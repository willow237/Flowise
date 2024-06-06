import {
    FlowiseWindowMemory,
    ICommonObject,
    IDatabaseEntity,
    IMessage,
    INode,
    INodeData,
    INodeParams,
    MemoryMethods
} from '../../../src/Interface'
import { getBaseClasses, mapChatMessageToBaseMessage } from '../../../src/utils'
import { BufferWindowMemory, BufferWindowMemoryInput } from 'langchain/memory'
import { BaseMessage } from '@langchain/core/messages'
import { DataSource } from 'typeorm'

class BufferWindowMemory_Memory implements INode {
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
        this.label = '缓冲窗口内存'
        this.name = 'bufferWindowMemory'
        this.version = 2.0
        this.type = 'BufferWindowMemory'
        this.icon = 'memory.svg'
        this.category = '记忆存储'
        this.description = '使用大小为 k 的窗口来显示最后 k 个来回，作为内存使用'
        this.baseClasses = [this.type, ...getBaseClasses(BufferWindowMemory)]
        this.inputs = [
            {
                label: '大小',
                name: 'k',
                type: 'number',
                default: '4',
                description: '大小为 k 的窗口，用于显示最后 k 个来回作为内存使用。'
            },
            {
                label: 'Session Id',
                name: 'sessionId',
                type: 'string',
                description:
                    '如果未指定，将使用随机 ID。 了解 <a target="_blank" href="https://docs.flowiseai.com/memory#ui-and-embedded-chat">更多</a>',
                default: '',
                optional: true,
                additionalParams: true
            },
            {
                label: '记忆存储键',
                name: 'memoryKey',
                type: 'string',
                default: 'chat_history',
                additionalParams: true
            }
        ]
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const k = nodeData.inputs?.k as string
        const sessionId = nodeData.inputs?.sessionId as string
        const memoryKey = (nodeData.inputs?.memoryKey as string) ?? 'chat_history'

        const appDataSource = options.appDataSource as DataSource
        const databaseEntities = options.databaseEntities as IDatabaseEntity
        const chatflowid = options.chatflowid as string

        const obj: Partial<BufferWindowMemoryInput> & BufferMemoryExtendedInput = {
            returnMessages: true,
            sessionId,
            memoryKey,
            k: parseInt(k, 10),
            appDataSource,
            databaseEntities,
            chatflowid
        }

        return new BufferWindowMemoryExtended(obj)
    }
}

interface BufferMemoryExtendedInput {
    sessionId: string
    appDataSource: DataSource
    databaseEntities: IDatabaseEntity
    chatflowid: string
}

class BufferWindowMemoryExtended extends FlowiseWindowMemory implements MemoryMethods {
    appDataSource: DataSource
    databaseEntities: IDatabaseEntity
    chatflowid: string
    sessionId = ''

    constructor(fields: BufferWindowMemoryInput & BufferMemoryExtendedInput) {
        super(fields)
        this.sessionId = fields.sessionId
        this.appDataSource = fields.appDataSource
        this.databaseEntities = fields.databaseEntities
        this.chatflowid = fields.chatflowid
    }

    async getChatMessages(
        overrideSessionId = '',
        returnBaseMessages = false,
        prependMessages?: IMessage[]
    ): Promise<IMessage[] | BaseMessage[]> {
        const id = overrideSessionId ? overrideSessionId : this.sessionId
        if (!id) return []

        let chatMessage = await this.appDataSource.getRepository(this.databaseEntities['ChatMessage']).find({
            where: {
                sessionId: id,
                chatflowid: this.chatflowid
            },
            take: this.k + 1,
            order: {
                createdDate: 'DESC' // we get the latest top K
            }
        })

        // reverse the order of human and ai messages
        if (chatMessage.length) chatMessage.reverse()

        if (prependMessages?.length) {
            chatMessage.unshift(...prependMessages)
        }

        if (returnBaseMessages) {
            return mapChatMessageToBaseMessage(chatMessage)
        }

        let returnIMessages: IMessage[] = []
        for (const m of chatMessage) {
            returnIMessages.push({
                message: m.content as string,
                type: m.role
            })
        }
        return returnIMessages
    }

    async addChatMessages(): Promise<void> {
        // adding chat messages is done on server level
        return
    }

    async clearChatMessages(): Promise<void> {
        // clearing chat messages is done on server level
        return
    }
}

module.exports = { nodeClass: BufferWindowMemory_Memory }
