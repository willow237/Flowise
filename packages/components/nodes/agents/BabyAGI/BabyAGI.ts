import { BaseChatModel } from '@langchain/core/language_models/chat_models'
import { VectorStore } from '@langchain/core/vectorstores'
import { INode, INodeData, INodeParams } from '../../../src/Interface'
import { BabyAGI } from './core'
import { checkInputs, Moderation } from '../../moderation/Moderation'
import { formatResponse } from '../../outputparsers/OutputParserHelpers'

class BabyAGI_Agents implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    inputs: INodeParams[]

    constructor() {
        this.label = 'BabyAGI'
        this.name = 'babyAGI'
        this.version = 2.0
        this.type = 'BabyAGI'
        this.category = '智能体'
        this.icon = 'babyagi.svg'
        this.description = '任务驱动型自主智能体，可根据目标创建新任务并调整任务列表的优先次序'
        this.baseClasses = ['BabyAGI']
        this.inputs = [
            {
                label: '聊天模型',
                name: 'model',
                type: 'BaseChatModel'
            },
            {
                label: '向量存储',
                name: 'vectorStore',
                type: 'VectorStore'
            },
            {
                label: '任务循环',
                name: 'taskLoop',
                type: 'number',
                default: 3
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

    async init(nodeData: INodeData): Promise<any> {
        const model = nodeData.inputs?.model as BaseChatModel
        const vectorStore = nodeData.inputs?.vectorStore as VectorStore
        const taskLoop = nodeData.inputs?.taskLoop as string
        const k = (vectorStore as any)?.k ?? 4

        const babyAgi = BabyAGI.fromLLM(model, vectorStore, parseInt(taskLoop, 10), k)
        return babyAgi
    }

    async run(nodeData: INodeData, input: string): Promise<string | object> {
        const executor = nodeData.instance as BabyAGI
        const moderations = nodeData.inputs?.inputModeration as Moderation[]

        if (moderations && moderations.length > 0) {
            try {
                // Use the output of the moderation chain as input for the BabyAGI agent
                input = await checkInputs(moderations, input)
            } catch (e) {
                await new Promise((resolve) => setTimeout(resolve, 500))
                //streamResponse(options.socketIO && options.socketIOClientId, e.message, options.socketIO, options.socketIOClientId)
                return formatResponse(e.message)
            }
        }

        const objective = input

        const res = await executor.call({ objective })
        return res
    }
}

module.exports = { nodeClass: BabyAGI_Agents }
