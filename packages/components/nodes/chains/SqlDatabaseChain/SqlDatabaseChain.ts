import { DataSourceOptions } from 'typeorm/data-source'
import { DataSource } from 'typeorm'
import { BaseLanguageModel } from '@langchain/core/language_models/base'
import { PromptTemplate, PromptTemplateInput } from '@langchain/core/prompts'
import { SqlDatabaseChain, SqlDatabaseChainInput, DEFAULT_SQL_DATABASE_PROMPT } from 'langchain/chains/sql_db'
import { SqlDatabase } from 'langchain/sql_db'
import { ICommonObject, INode, INodeData, INodeParams } from '../../../src/Interface'
import { ConsoleCallbackHandler, CustomChainHandler, additionalCallbacks } from '../../../src/handler'
import { getBaseClasses, getInputVariables } from '../../../src/utils'
import { checkInputs, Moderation, streamResponse } from '../../moderation/Moderation'
import { formatResponse } from '../../outputparsers/OutputParserHelpers'

type DatabaseType = 'sqlite' | 'postgres' | 'mssql' | 'mysql'

class SqlDatabaseChain_Chains implements INode {
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
        this.label = 'Sql 数据库对话链'
        this.name = 'sqlDatabaseChain'
        this.version = 5.0
        this.type = 'SqlDatabaseChain'
        this.icon = 'sqlchain.svg'
        this.category = '对话链'
        this.description = '通过 SQL 数据库回答问题'
        this.baseClasses = [this.type, ...getBaseClasses(SqlDatabaseChain)]
        this.inputs = [
            {
                label: '语言模型',
                name: 'model',
                type: 'BaseLanguageModel'
            },
            {
                label: '数据库',
                name: 'database',
                type: 'options',
                options: [
                    {
                        label: 'SQLite',
                        name: 'sqlite'
                    },
                    {
                        label: 'PostgreSQL',
                        name: 'postgres'
                    },
                    {
                        label: 'MSSQL',
                        name: 'mssql'
                    },
                    {
                        label: 'MySQL',
                        name: 'mysql'
                    }
                ],
                default: 'sqlite'
            },
            {
                label: '连接字符串或文件路径（仅限 sqlite）',
                name: 'url',
                type: 'string',
                placeholder: '1270.0.0.1:5432/chinook'
            },
            {
                label: '包含的表',
                name: 'includesTables',
                type: 'string',
                description: '查询要包含的表，以逗号分隔。只能使用包含表或忽略表',
                placeholder: 'table1, table2',
                additionalParams: true,
                optional: true
            },
            {
                label: '忽略的表',
                name: 'ignoreTables',
                type: 'string',
                description: '查询时要忽略的表，用逗号分隔。只能使用忽略表或包含表',
                placeholder: 'table1, table2',
                additionalParams: true,
                optional: true
            },
            {
                label: '示例表的行信息',
                name: 'sampleRowsInTableInfo',
                type: 'number',
                description: '表中要加载的信息样本行数。',
                placeholder: '3',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Top Keys',
                name: 'topK',
                type: 'number',
                description:
                    '如果你要查询表中的多行数据，可以使用“top_k”参数（默认值为10）来选择要获取的最大结果数。这很有用，可以避免查询结果超过提示的最大长度或不必要地消耗令牌。',
                placeholder: '10',
                additionalParams: true,
                optional: true
            },
            {
                label: '自定义 Prompt',
                name: 'customPrompt',
                type: 'string',
                description:
                    '你可以为链提供自定义提示。这将覆盖现有的默认提示。参见<a target="_blank" href="https://python.langchain.com/docs/integrations/tools/sqlite#customize-prompt">指南</a>',
                warning: '提示必须包含 3 个输入变量： {input}、{dialect}、{table_info}。您可以参考上述说明中的官方指南',
                rows: 4,
                placeholder: DEFAULT_SQL_DATABASE_PROMPT.template + DEFAULT_SQL_DATABASE_PROMPT.templateFormat,
                additionalParams: true,
                optional: true
            },
            {
                label: '输入调节',
                description: '检测可能产生有害输出的文本，防止将其发送给语言模型',
                name: 'inputModeration',
                type: 'Moderation',
                optional: true,
                list: true
            }
        ]
    }

    async init(nodeData: INodeData): Promise<any> {
        const databaseType = nodeData.inputs?.database as DatabaseType
        const model = nodeData.inputs?.model as BaseLanguageModel
        const url = nodeData.inputs?.url as string
        const includesTables = nodeData.inputs?.includesTables
        const splittedIncludesTables = includesTables == '' ? undefined : includesTables?.split(',')
        const ignoreTables = nodeData.inputs?.ignoreTables
        const splittedIgnoreTables = ignoreTables == '' ? undefined : ignoreTables?.split(',')
        const sampleRowsInTableInfo = nodeData.inputs?.sampleRowsInTableInfo as number
        const topK = nodeData.inputs?.topK as number
        const customPrompt = nodeData.inputs?.customPrompt as string

        const chain = await getSQLDBChain(
            databaseType,
            url,
            model,
            splittedIncludesTables,
            splittedIgnoreTables,
            sampleRowsInTableInfo,
            topK,
            customPrompt
        )
        return chain
    }

    async run(nodeData: INodeData, input: string, options: ICommonObject): Promise<string | object> {
        const databaseType = nodeData.inputs?.database as DatabaseType
        const model = nodeData.inputs?.model as BaseLanguageModel
        const url = nodeData.inputs?.url as string
        const includesTables = nodeData.inputs?.includesTables
        const splittedIncludesTables = includesTables == '' ? undefined : includesTables?.split(',')
        const ignoreTables = nodeData.inputs?.ignoreTables
        const splittedIgnoreTables = ignoreTables == '' ? undefined : ignoreTables?.split(',')
        const sampleRowsInTableInfo = nodeData.inputs?.sampleRowsInTableInfo as number
        const topK = nodeData.inputs?.topK as number
        const customPrompt = nodeData.inputs?.customPrompt as string
        const moderations = nodeData.inputs?.inputModeration as Moderation[]
        if (moderations && moderations.length > 0) {
            try {
                // Use the output of the moderation chain as input for the Sql Database Chain
                input = await checkInputs(moderations, input)
            } catch (e) {
                await new Promise((resolve) => setTimeout(resolve, 500))
                streamResponse(options.socketIO && options.socketIOClientId, e.message, options.socketIO, options.socketIOClientId)
                return formatResponse(e.message)
            }
        }

        const chain = await getSQLDBChain(
            databaseType,
            url,
            model,
            splittedIncludesTables,
            splittedIgnoreTables,
            sampleRowsInTableInfo,
            topK,
            customPrompt
        )
        const loggerHandler = new ConsoleCallbackHandler(options.logger)
        const callbacks = await additionalCallbacks(nodeData, options)

        if (options.socketIO && options.socketIOClientId) {
            const handler = new CustomChainHandler(options.socketIO, options.socketIOClientId, 2)
            const res = await chain.run(input, [loggerHandler, handler, ...callbacks])
            return res
        } else {
            const res = await chain.run(input, [loggerHandler, ...callbacks])
            return res
        }
    }
}

const getSQLDBChain = async (
    databaseType: DatabaseType,
    url: string,
    llm: BaseLanguageModel,
    includesTables?: string[],
    ignoreTables?: string[],
    sampleRowsInTableInfo?: number,
    topK?: number,
    customPrompt?: string
) => {
    const datasource = new DataSource(
        databaseType === 'sqlite'
            ? {
                  type: databaseType,
                  database: url
              }
            : ({
                  type: databaseType,
                  url: url
              } as DataSourceOptions)
    )

    const db = await SqlDatabase.fromDataSourceParams({
        appDataSource: datasource,
        includesTables: includesTables,
        ignoreTables: ignoreTables,
        sampleRowsInTableInfo: sampleRowsInTableInfo
    })

    const obj: SqlDatabaseChainInput = {
        llm,
        database: db,
        verbose: process.env.DEBUG === 'true' ? true : false,
        topK: topK
    }

    if (customPrompt) {
        const options: PromptTemplateInput = {
            template: customPrompt,
            inputVariables: getInputVariables(customPrompt)
        }
        obj.prompt = new PromptTemplate(options)
    }

    const chain = new SqlDatabaseChain(obj)
    return chain
}

module.exports = { nodeClass: SqlDatabaseChain_Chains }
