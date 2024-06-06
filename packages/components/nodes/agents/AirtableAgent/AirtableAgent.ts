import axios from 'axios'
import { BaseLanguageModel } from '@langchain/core/language_models/base'
import { AgentExecutor } from 'langchain/agents'
import { LLMChain } from 'langchain/chains'
import { ICommonObject, INode, INodeData, INodeParams, PromptTemplate } from '../../../src/Interface'
import { getBaseClasses, getCredentialData, getCredentialParam } from '../../../src/utils'
import { ConsoleCallbackHandler, CustomChainHandler, additionalCallbacks } from '../../../src/handler'
import { LoadPyodide, finalSystemPrompt, systemPrompt } from './core'
import { checkInputs, Moderation } from '../../moderation/Moderation'
import { formatResponse } from '../../outputparsers/OutputParserHelpers'

class Airtable_Agents implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    credential: INodeParams
    inputs: INodeParams[]

    constructor() {
        this.label = 'Airtable 智能体'
        this.name = 'airtableAgent'
        this.version = 2.0
        this.type = 'AgentExecutor'
        this.category = '智能体'
        this.icon = 'airtable.svg'
        this.description = '用于回答 Airtable 表查询的智能体'
        this.baseClasses = [this.type, ...getBaseClasses(AgentExecutor)]
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['airtableApi']
        }
        this.inputs = [
            {
                label: '语言模型',
                name: 'model',
                type: 'BaseLanguageModel'
            },
            {
                label: 'Base Id',
                name: 'baseId',
                type: 'string',
                placeholder: 'app11RobdGoX0YNsC',
                description:
                    '如果您的表格 URL 是这样的：https://airtable.com/app11RobdGoX0YNsC/tblJdmvbrgizbYICO/viw9UrP77Id0CE4ee, app11RovdGoX0YNsC 是 base id'
            },
            {
                label: 'Table Id',
                name: 'tableId',
                type: 'string',
                placeholder: 'tblJdmvbrgizbYICO',
                description:
                    '如果您的表格 URL 是这样的：https://airtable.com/app11RobdGoX0YNsC/tblJdmvbrgizbYICO/viw9UrP77Id0CE4ee, tblJdmvbrgizbYICO 是 table id'
            },
            {
                label: '返回全部',
                name: 'returnAll',
                type: 'boolean',
                default: true,
                additionalParams: true,
                description: '是返回所有结果，还是只返回给定限制内的结果'
            },
            {
                label: '限制',
                name: 'limit',
                type: 'number',
                default: 100,
                additionalParams: true,
                description: '返回的结果数'
            },
            {
                label: '输入调节',
                description: '检测可能生成有害输出的文本并阻止其发送到语言模型',
                name: 'inputModeration',
                type: 'Moderation',
                optional: true,
                list: true
            }
        ]
    }

    async init(): Promise<any> {
        // Not used
        return undefined
    }

    async run(nodeData: INodeData, input: string, options: ICommonObject): Promise<string | object> {
        const model = nodeData.inputs?.model as BaseLanguageModel
        const baseId = nodeData.inputs?.baseId as string
        const tableId = nodeData.inputs?.tableId as string
        const returnAll = nodeData.inputs?.returnAll as boolean
        const limit = nodeData.inputs?.limit as string
        const moderations = nodeData.inputs?.inputModeration as Moderation[]

        if (moderations && moderations.length > 0) {
            try {
                // Use the output of the moderation chain as input for the Vectara chain
                input = await checkInputs(moderations, input)
            } catch (e) {
                await new Promise((resolve) => setTimeout(resolve, 500))
                //streamResponse(options.socketIO && options.socketIOClientId, e.message, options.socketIO, options.socketIOClientId)
                return formatResponse(e.message)
            }
        }

        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const accessToken = getCredentialParam('accessToken', credentialData, nodeData)

        let airtableData: ICommonObject[] = []

        if (returnAll) {
            airtableData = await loadAll(baseId, tableId, accessToken)
        } else {
            airtableData = await loadLimit(limit ? parseInt(limit, 10) : 100, baseId, tableId, accessToken)
        }

        let base64String = Buffer.from(JSON.stringify(airtableData)).toString('base64')

        const loggerHandler = new ConsoleCallbackHandler(options.logger)
        const handler = new CustomChainHandler(options.socketIO, options.socketIOClientId)
        const callbacks = await additionalCallbacks(nodeData, options)

        const pyodide = await LoadPyodide()

        // First load the csv file and get the dataframe dictionary of column types
        // For example using titanic.csv: {'PassengerId': 'int64', 'Survived': 'int64', 'Pclass': 'int64', 'Name': 'object', 'Sex': 'object', 'Age': 'float64', 'SibSp': 'int64', 'Parch': 'int64', 'Ticket': 'object', 'Fare': 'float64', 'Cabin': 'object', 'Embarked': 'object'}
        let dataframeColDict = ''
        try {
            const code = `import pandas as pd
import base64
import json

base64_string = "${base64String}"

decoded_data = base64.b64decode(base64_string)

json_data = json.loads(decoded_data)

df = pd.DataFrame(json_data)
my_dict = df.dtypes.astype(str).to_dict()
print(my_dict)
json.dumps(my_dict)`
            dataframeColDict = await pyodide.runPythonAsync(code)
        } catch (error) {
            throw new Error(error)
        }

        // Then tell GPT to come out with ONLY python code
        // For example: len(df), df[df['SibSp'] > 3]['PassengerId'].count()
        let pythonCode = ''
        if (dataframeColDict) {
            const chain = new LLMChain({
                llm: model,
                prompt: PromptTemplate.fromTemplate(systemPrompt),
                verbose: process.env.DEBUG === 'true' ? true : false
            })
            const inputs = {
                dict: dataframeColDict,
                question: input
            }
            const res = await chain.call(inputs, [loggerHandler, ...callbacks])
            pythonCode = res?.text
        }

        // Then run the code using Pyodide
        let finalResult = ''
        if (pythonCode) {
            try {
                const code = `import pandas as pd\n${pythonCode}`
                finalResult = await pyodide.runPythonAsync(code)
            } catch (error) {
                throw new Error(`Sorry, I'm unable to find answer for question: "${input}" using follwoing code: "${pythonCode}"`)
            }
        }

        // Finally, return a complete answer
        if (finalResult) {
            const chain = new LLMChain({
                llm: model,
                prompt: PromptTemplate.fromTemplate(finalSystemPrompt),
                verbose: process.env.DEBUG === 'true' ? true : false
            })
            const inputs = {
                question: input,
                answer: finalResult
            }

            if (options.socketIO && options.socketIOClientId) {
                const result = await chain.call(inputs, [loggerHandler, handler, ...callbacks])
                return result?.text
            } else {
                const result = await chain.call(inputs, [loggerHandler, ...callbacks])
                return result?.text
            }
        }

        return pythonCode
    }
}

interface AirtableLoaderResponse {
    records: AirtableLoaderPage[]
    offset?: string
}

interface AirtableLoaderPage {
    id: string
    createdTime: string
    fields: ICommonObject
}

const fetchAirtableData = async (url: string, params: ICommonObject, accessToken: string): Promise<AirtableLoaderResponse> => {
    try {
        const headers = {
            Authorization: `Bearer ${accessToken}`,
            'Content-Type': 'application/json',
            Accept: 'application/json'
        }
        const response = await axios.get(url, { params, headers })
        return response.data
    } catch (error) {
        throw new Error(`Failed to fetch ${url} from Airtable: ${error}`)
    }
}

const loadAll = async (baseId: string, tableId: string, accessToken: string): Promise<ICommonObject[]> => {
    const params: ICommonObject = { pageSize: 100 }
    let data: AirtableLoaderResponse
    let returnPages: AirtableLoaderPage[] = []

    do {
        data = await fetchAirtableData(`https://api.airtable.com/v0/${baseId}/${tableId}`, params, accessToken)
        returnPages.push.apply(returnPages, data.records)
        params.offset = data.offset
    } while (data.offset !== undefined)

    return data.records.map((page) => page.fields)
}

const loadLimit = async (limit: number, baseId: string, tableId: string, accessToken: string): Promise<ICommonObject[]> => {
    const params = { maxRecords: limit }
    const data = await fetchAirtableData(`https://api.airtable.com/v0/${baseId}/${tableId}`, params, accessToken)
    if (data.records.length === 0) {
        return []
    }
    return data.records.map((page) => page.fields)
}

module.exports = { nodeClass: Airtable_Agents }
