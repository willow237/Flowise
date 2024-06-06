import { flatten } from 'lodash'
import { ChatMessage, OpenAI, OpenAIAgent } from 'llamaindex'
import { getBaseClasses } from '../../../../src/utils'
import { FlowiseMemory, ICommonObject, IMessage, INode, INodeData, INodeParams, IUsedTool } from '../../../../src/Interface'

class OpenAIFunctionAgent_LlamaIndex_Agents implements INode {
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
        this.label = 'OpenAI 工具智能体'
        this.name = 'openAIToolAgentLlamaIndex'
        this.version = 2.0
        this.type = 'OpenAIToolAgent'
        this.category = '智能体'
        this.icon = 'function.svg'
        this.description = `使用 OpenAI 函数调用的智能体，利用 LlamaIndex 挑选要调用的工具和参数`
        this.baseClasses = [this.type, ...getBaseClasses(OpenAIAgent)]
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
                label: 'OpenAI/Azure 聊天模型',
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
        const model = nodeData.inputs?.model as OpenAI
        const systemMessage = nodeData.inputs?.systemMessage as string
        let tools = nodeData.inputs?.tools
        tools = flatten(tools)

        const isStreamingEnabled = options.socketIO && options.socketIOClientId

        const chatHistory = [] as ChatMessage[]

        if (systemMessage) {
            chatHistory.push({
                content: systemMessage,
                role: 'system'
            })
        }

        const msgs = (await memory.getChatMessages(this.sessionId, false)) as IMessage[]
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

        const agent = new OpenAIAgent({
            tools,
            llm: model,
            chatHistory: chatHistory,
            verbose: process.env.DEBUG === 'true' ? true : false
        })

        let text = ''
        let isStreamingStarted = false
        const usedTools: IUsedTool[] = []

        if (isStreamingEnabled) {
            const stream = await agent.chat({
                message: input,
                chatHistory,
                stream: true,
                verbose: process.env.DEBUG === 'true' ? true : false
            })
            for await (const chunk of stream) {
                //console.log('chunk', chunk)
                text += chunk.response.delta
                if (!isStreamingStarted) {
                    isStreamingStarted = true
                    options.socketIO.to(options.socketIOClientId).emit('start', chunk.response.delta)
                    if (chunk.sources.length) {
                        for (const sourceTool of chunk.sources) {
                            usedTools.push({
                                tool: sourceTool.tool?.metadata.name ?? '',
                                toolInput: sourceTool.input,
                                toolOutput: sourceTool.output as any
                            })
                        }
                        options.socketIO.to(options.socketIOClientId).emit('usedTools', usedTools)
                    }
                }

                options.socketIO.to(options.socketIOClientId).emit('token', chunk.response.delta)
            }
        } else {
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

module.exports = { nodeClass: OpenAIFunctionAgent_LlamaIndex_Agents }
