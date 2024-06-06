import { flatten } from 'lodash'
import { MessageContentTextDetail, ChatMessage, AnthropicAgent, Anthropic } from 'llamaindex'
import { getBaseClasses } from '../../../../src/utils'
import { FlowiseMemory, ICommonObject, IMessage, INode, INodeData, INodeParams, IUsedTool } from '../../../../src/Interface'

class AnthropicAgent_LlamaIndex_Agents implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    tags: string[]
    inputs: INodeParams[]
    sessionId?: string
    badge?: string

    constructor(fields?: { sessionId?: string }) {
        this.label = 'Anthropic 智能体'
        this.name = 'anthropicAgentLlamaIndex'
        this.version = 1.0
        this.type = 'AnthropicAgent'
        this.category = '智能体'
        this.icon = 'Anthropic.svg'
        this.description = `使用 Anthropic Claude 函数调用的智能体，利用 LlamaIndex 挑选要调用的工具和参数`
        this.baseClasses = [this.type, ...getBaseClasses(AnthropicAgent)]
        this.tags = ['LlamaIndex']
        this.inputs = [
            {
                label: '工具',
                name: 'tools',
                type: 'Tool_LlamaIndex',
                list: true
            },
            {
                label: '记忆存储',
                name: 'memory',
                type: 'BaseChatMemory'
            },
            {
                label: 'Anthropic Claude 模型',
                name: 'model',
                type: 'BaseChatModel_LlamaIndex'
            },
            {
                label: '系统信息',
                name: 'systemMessage',
                type: 'string',
                rows: 4,
                optional: true,
                additionalParams: true
            }
        ]
        this.sessionId = fields?.sessionId
    }

    async init(): Promise<any> {
        return null
    }

    async run(nodeData: INodeData, input: string, options: ICommonObject): Promise<string | ICommonObject> {
        const memory = nodeData.inputs?.memory as FlowiseMemory
        const model = nodeData.inputs?.model as Anthropic
        const systemMessage = nodeData.inputs?.systemMessage as string
        const prependMessages = options?.prependMessages

        let tools = nodeData.inputs?.tools
        tools = flatten(tools)

        const chatHistory = [] as ChatMessage[]

        if (systemMessage) {
            chatHistory.push({
                content: systemMessage,
                role: 'system'
            })
        }

        const msgs = (await memory.getChatMessages(this.sessionId, false, prependMessages)) as IMessage[]
        for (const message of msgs) {
            if (message.type === 'apiMessage') {
                chatHistory.push({
                    content: message.message,
                    role: 'assistant'
                })
            } else if (message.type === 'userMessage') {
                chatHistory.push({
                    content: message.message,
                    role: 'user'
                })
            }
        }

        const agent = new AnthropicAgent({
            tools,
            llm: model,
            chatHistory: chatHistory,
            verbose: process.env.DEBUG === 'true' ? true : false
        })

        let text = ''
        const usedTools: IUsedTool[] = []

        const response = await agent.chat({ message: input, chatHistory, verbose: process.env.DEBUG === 'true' ? true : false })

        if (response.sources.length) {
            for (const sourceTool of response.sources) {
                usedTools.push({
                    tool: sourceTool.tool?.metadata.name ?? '',
                    toolInput: sourceTool.input,
                    toolOutput: sourceTool.output as any
                })
            }
        }

        if (Array.isArray(response.response.message.content) && response.response.message.content.length > 0) {
            text = (response.response.message.content[0] as MessageContentTextDetail).text
        } else {
            text = response.response.message.content as string
        }

        await memory.addChatMessages(
            [
                {
                    text: input,
                    type: 'userMessage'
                },
                {
                    text: text,
                    type: 'apiMessage'
                }
            ],
            this.sessionId
        )

        return usedTools.length ? { text: text, usedTools } : text
    }
}

module.exports = { nodeClass: AnthropicAgent_LlamaIndex_Agents }
