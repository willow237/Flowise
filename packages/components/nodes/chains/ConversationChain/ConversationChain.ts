import { ConversationChain } from 'langchain/chains'
import {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    BaseMessagePromptTemplateLike,
    PromptTemplate
} from '@langchain/core/prompts'
import { RunnableSequence } from '@langchain/core/runnables'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { BaseChatModel } from '@langchain/core/language_models/chat_models'
import { HumanMessage } from '@langchain/core/messages'
import { ConsoleCallbackHandler as LCConsoleCallbackHandler } from '@langchain/core/tracers/console'
import { checkInputs, Moderation, streamResponse } from '../../moderation/Moderation'
import { formatResponse } from '../../outputparsers/OutputParserHelpers'
import { addImagesToMessages, llmSupportsVision } from '../../../src/multiModalUtils'
import { ChatOpenAI } from '../../chatmodels/ChatOpenAI/FlowiseChatOpenAI'
import {
    IVisionChatModal,
    FlowiseMemory,
    ICommonObject,
    INode,
    INodeData,
    INodeParams,
    MessageContentImageUrl
} from '../../../src/Interface'
import { ConsoleCallbackHandler, CustomChainHandler, additionalCallbacks } from '../../../src/handler'
import { getBaseClasses, handleEscapeCharacters } from '../../../src/utils'

let systemMessage = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.`
const inputKey = 'input'

class ConversationChain_Chains implements INode {
    label: string
    name: string
    version: number
    type: string
    icon: string
    category: string
    baseClasses: string[]
    description: string
    inputs: INodeParams[]
    sessionId?: string

    constructor(fields?: { sessionId?: string }) {
        this.label = '对话链'
        this.name = 'conversationChain'
        this.version = 3.0
        this.type = 'ConversationChain'
        this.icon = 'conv.svg'
        this.category = '对话链'
        this.description = '带有记忆功能的聊天模型特定对话链'
        this.baseClasses = [this.type, ...getBaseClasses(ConversationChain)]
        this.inputs = [
            {
                label: '聊天模型',
                name: 'model',
                type: 'BaseChatModel'
            },
            {
                label: '记忆存储',
                name: 'memory',
                type: 'BaseMemory'
            },
            {
                label: '聊天提示模板',
                name: 'chatPromptTemplate',
                type: 'ChatPromptTemplate',
                description: '使用聊天提示模板覆盖现有提示。人类消息必须包括 {input} 变量',
                optional: true
            },
            /* Deprecated
            {
                label: 'Document',
                name: 'document',
                type: 'Document',
                description:
                    'Include whole document into the context window, if you get maximum context length error, please use model with higher context window like Claude 100k, or gpt4 32k',
                optional: true,
                list: true
            },*/
            {
                label: '输入调节',
                description: '检测可能生成有害输出的文本并阻止其发送到语言模型',
                name: 'inputModeration',
                type: 'Moderation',
                optional: true,
                list: true
            },
            {
                label: '系统信息',
                name: 'systemMessagePrompt',
                type: 'string',
                rows: 4,
                description: '如果提供了聊天提示模板，则将忽略此提示',
                additionalParams: true,
                optional: true,
                default: systemMessage,
                placeholder: systemMessage
            }
        ]
        this.sessionId = fields?.sessionId
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const chain = prepareChain(nodeData, options, this.sessionId)
        return chain
    }

    async run(nodeData: INodeData, input: string, options: ICommonObject): Promise<string | object> {
        const memory = nodeData.inputs?.memory

        const chain = await prepareChain(nodeData, options, this.sessionId)
        const moderations = nodeData.inputs?.inputModeration as Moderation[]

        if (moderations && moderations.length > 0) {
            try {
                // Use the output of the moderation chain as input for the LLM chain
                input = await checkInputs(moderations, input)
            } catch (e) {
                await new Promise((resolve) => setTimeout(resolve, 500))
                streamResponse(options.socketIO && options.socketIOClientId, e.message, options.socketIO, options.socketIOClientId)
                return formatResponse(e.message)
            }
        }

        const loggerHandler = new ConsoleCallbackHandler(options.logger)
        const additionalCallback = await additionalCallbacks(nodeData, options)

        let res = ''
        let callbacks = [loggerHandler, ...additionalCallback]

        if (process.env.DEBUG === 'true') {
            callbacks.push(new LCConsoleCallbackHandler())
        }

        if (options.socketIO && options.socketIOClientId) {
            const handler = new CustomChainHandler(options.socketIO, options.socketIOClientId)
            callbacks.push(handler)
            res = await chain.invoke({ input }, { callbacks })
        } else {
            res = await chain.invoke({ input }, { callbacks })
        }

        await memory.addChatMessages(
            [
                {
                    text: input,
                    type: 'userMessage'
                },
                {
                    text: res,
                    type: 'apiMessage'
                }
            ],
            this.sessionId
        )

        return res
    }
}

const prepareChatPrompt = (nodeData: INodeData, humanImageMessages: MessageContentImageUrl[]) => {
    const memory = nodeData.inputs?.memory as FlowiseMemory
    const prompt = nodeData.inputs?.systemMessagePrompt as string
    const chatPromptTemplate = nodeData.inputs?.chatPromptTemplate as ChatPromptTemplate
    let model = nodeData.inputs?.model as BaseChatModel

    if (chatPromptTemplate && chatPromptTemplate.promptMessages.length) {
        const sysPrompt = chatPromptTemplate.promptMessages[0]
        const humanPrompt = chatPromptTemplate.promptMessages[chatPromptTemplate.promptMessages.length - 1]
        const messages = [sysPrompt, new MessagesPlaceholder(memory.memoryKey ?? 'chat_history'), humanPrompt]

        // OpenAI works better when separate images into standalone human messages
        if (model instanceof ChatOpenAI && humanImageMessages.length) {
            messages.push(new HumanMessage({ content: [...humanImageMessages] }))
        } else if (humanImageMessages.length) {
            const lastMessage = messages.pop() as HumanMessagePromptTemplate
            const template = (lastMessage.prompt as PromptTemplate).template as string
            const msg = HumanMessagePromptTemplate.fromTemplate([
                ...humanImageMessages,
                {
                    text: template
                }
            ])
            msg.inputVariables = lastMessage.inputVariables
            messages.push(msg)
        }

        const chatPrompt = ChatPromptTemplate.fromMessages(messages)
        if ((chatPromptTemplate as any).promptValues) {
            // @ts-ignore
            chatPrompt.promptValues = (chatPromptTemplate as any).promptValues
        }

        return chatPrompt
    }

    const messages: BaseMessagePromptTemplateLike[] = [
        SystemMessagePromptTemplate.fromTemplate(prompt ? prompt : systemMessage),
        new MessagesPlaceholder(memory.memoryKey ?? 'chat_history'),
        HumanMessagePromptTemplate.fromTemplate(`{${inputKey}}`)
    ]

    // OpenAI works better when separate images into standalone human messages
    if (model instanceof ChatOpenAI && humanImageMessages.length) {
        messages.push(new HumanMessage({ content: [...humanImageMessages] }))
    } else if (humanImageMessages.length) {
        messages.pop()
        messages.push(HumanMessagePromptTemplate.fromTemplate([`{${inputKey}}`, ...humanImageMessages]))
    }

    const chatPrompt = ChatPromptTemplate.fromMessages(messages)

    return chatPrompt
}

const prepareChain = async (nodeData: INodeData, options: ICommonObject, sessionId?: string) => {
    let model = nodeData.inputs?.model as BaseChatModel
    const memory = nodeData.inputs?.memory as FlowiseMemory
    const memoryKey = memory.memoryKey ?? 'chat_history'
    const prependMessages = options?.prependMessages

    let messageContent: MessageContentImageUrl[] = []
    if (llmSupportsVision(model)) {
        messageContent = await addImagesToMessages(nodeData, options, model.multiModalOption)
        const visionChatModel = model as IVisionChatModal
        if (messageContent?.length) {
            visionChatModel.setVisionModel()
        } else {
            // revert to previous values if image upload is empty
            visionChatModel.revertToOriginalModel()
        }
    }

    const chatPrompt = prepareChatPrompt(nodeData, messageContent)
    let promptVariables = {}
    const promptValuesRaw = (chatPrompt as any).promptValues
    if (promptValuesRaw) {
        const promptValues = handleEscapeCharacters(promptValuesRaw, true)
        for (const val in promptValues) {
            promptVariables = {
                ...promptVariables,
                [val]: () => {
                    return promptValues[val]
                }
            }
        }
    }

    const conversationChain = RunnableSequence.from([
        {
            [inputKey]: (input: { input: string }) => input.input,
            [memoryKey]: async () => {
                const history = await memory.getChatMessages(sessionId, true, prependMessages)
                return history
            },
            ...promptVariables
        },
        prepareChatPrompt(nodeData, messageContent),
        model,
        new StringOutputParser()
    ])

    return conversationChain
}

module.exports = { nodeClass: ConversationChain_Chains }
