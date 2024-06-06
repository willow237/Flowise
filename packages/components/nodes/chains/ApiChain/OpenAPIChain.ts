import { ChatOpenAI } from '@langchain/openai'
import { APIChain, createOpenAPIChain } from 'langchain/chains'
import { ICommonObject, INode, INodeData, INodeParams } from '../../../src/Interface'
import { getBaseClasses } from '../../../src/utils'
import { ConsoleCallbackHandler, CustomChainHandler, additionalCallbacks } from '../../../src/handler'
import { checkInputs, Moderation, streamResponse } from '../../moderation/Moderation'
import { formatResponse } from '../../outputparsers/OutputParserHelpers'
import { getFileFromStorage } from '../../../src'

class OpenApiChain_Chains implements INode {
    label: string
    name: string
    version: number
    type: string
    icon: string
    category: string
    baseClasses: string[]
    description: string
    inputs: INodeParams[]

    constructor() {
        this.label = 'OpenAPI 对话链'
        this.name = 'openApiChain'
        this.version = 2.0
        this.type = 'OpenAPIChain'
        this.icon = 'openapi.svg'
        this.category = '对话链'
        this.description = '仅根据 OpenAPI 规范自动选择和调用 API 的链'
        this.baseClasses = [this.type, ...getBaseClasses(APIChain)]
        this.inputs = [
            {
                label: 'ChatOpenAI 模型',
                name: 'model',
                type: 'ChatOpenAI'
            },
            {
                label: 'YAML 链接',
                name: 'yamlLink',
                type: 'string',
                placeholder: 'https://api.speak.com/openapi.yaml',
                description: '如果提供了 YAML 链接，上传的 YAML 文件将被忽略，而使用 YAML 链接'
            },
            {
                label: 'YAML 文件',
                name: 'yamlFile',
                type: 'file',
                fileType: '.yaml',
                description: 'I如果提供了 YAML 链接，上传的 YAML 文件将被忽略，而使用 YAML 链接'
            },
            {
                label: 'Headers',
                name: 'headers',
                type: 'json',
                additionalParams: true,
                optional: true
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

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        return await initChain(nodeData, options)
    }

    async run(nodeData: INodeData, input: string, options: ICommonObject): Promise<string | object> {
        const chain = await initChain(nodeData, options)
        const loggerHandler = new ConsoleCallbackHandler(options.logger)
        const callbacks = await additionalCallbacks(nodeData, options)
        const moderations = nodeData.inputs?.inputModeration as Moderation[]
        if (moderations && moderations.length > 0) {
            try {
                // Use the output of the moderation chain as input for the OpenAPI chain
                input = await checkInputs(moderations, input)
            } catch (e) {
                await new Promise((resolve) => setTimeout(resolve, 500))
                streamResponse(options.socketIO && options.socketIOClientId, e.message, options.socketIO, options.socketIOClientId)
                return formatResponse(e.message)
            }
        }
        if (options.socketIO && options.socketIOClientId) {
            const handler = new CustomChainHandler(options.socketIO, options.socketIOClientId)
            const res = await chain.run(input, [loggerHandler, handler, ...callbacks])
            return res
        } else {
            const res = await chain.run(input, [loggerHandler, ...callbacks])
            return res
        }
    }
}

const initChain = async (nodeData: INodeData, options: ICommonObject) => {
    const model = nodeData.inputs?.model as ChatOpenAI
    const headers = nodeData.inputs?.headers as string
    const yamlLink = nodeData.inputs?.yamlLink as string
    const yamlFileBase64 = nodeData.inputs?.yamlFile as string

    let yamlString = ''

    if (yamlLink) {
        yamlString = yamlLink
    } else {
        if (yamlFileBase64.startsWith('FILE-STORAGE::')) {
            const file = yamlFileBase64.replace('FILE-STORAGE::', '')
            const chatflowid = options.chatflowid
            const fileData = await getFileFromStorage(file, chatflowid)
            yamlString = fileData.toString()
        } else {
            const splitDataURI = yamlFileBase64.split(',')
            splitDataURI.pop()
            const bf = Buffer.from(splitDataURI.pop() || '', 'base64')
            yamlString = bf.toString('utf-8')
        }
    }

    return await createOpenAPIChain(yamlString, {
        llm: model,
        headers: typeof headers === 'object' ? headers : headers ? JSON.parse(headers) : {},
        verbose: process.env.DEBUG === 'true' ? true : false
    })
}

module.exports = { nodeClass: OpenApiChain_Chains }
