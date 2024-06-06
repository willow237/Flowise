import { Moderation } from '../Moderation'
import { OpenAIModerationRunner } from './OpenAIModerationRunner'
import { ICommonObject, INode, INodeData, INodeParams } from '../../../src/Interface'
import { getBaseClasses, getCredentialData, getCredentialParam } from '../../../src'

class OpenAIModeration implements INode {
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
        this.label = 'OpenAI 内容审核'
        this.name = 'inputModerationOpenAI'
        this.version = 1.0
        this.type = 'Moderation'
        this.icon = 'openai.svg'
        this.category = '内容审核'
        this.description = '检查内容是否符合 OpenAI 使用政策。'
        this.baseClasses = [this.type, ...getBaseClasses(Moderation)]
        this.credential = {
            label: '连接凭据',
            name: 'credential',
            type: 'credential',
            credentialNames: ['openAIApi']
        }
        this.inputs = [
            {
                label: '错误信息',
                name: 'moderationErrorMessage',
                type: 'string',
                rows: 2,
                default: '无法处理！输入违反了 OpenAI 的内容审核政策。',
                optional: true
            }
        ]
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const openAIApiKey = getCredentialParam('openAIApiKey', credentialData, nodeData)

        const runner = new OpenAIModerationRunner(openAIApiKey)
        const moderationErrorMessage = nodeData.inputs?.moderationErrorMessage as string
        if (moderationErrorMessage) runner.setErrorMessage(moderationErrorMessage)
        return runner
    }
}

module.exports = { nodeClass: OpenAIModeration }
