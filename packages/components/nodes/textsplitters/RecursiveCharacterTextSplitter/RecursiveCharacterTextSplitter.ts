import { INode, INodeData, INodeParams } from '../../../src/Interface'
import { getBaseClasses } from '../../../src/utils'
import { RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitterParams } from 'langchain/text_splitter'

class RecursiveCharacterTextSplitter_TextSplitters implements INode {
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
        this.label = '递归字符文本分割器'
        this.name = 'recursiveCharacterTextSplitter'
        this.version = 2.0
        this.type = 'RecursiveCharacterTextSplitter'
        this.icon = 'textsplitter.svg'
        this.category = '文本分割器'
        this.description = `按不同字符递归分割文档--从“\\n\n ”开始，然后是“\\n”，然后是“ ”`
        this.baseClasses = [this.type, ...getBaseClasses(RecursiveCharacterTextSplitter)]
        this.inputs = [
            {
                label: '块大小',
                name: 'chunkSize',
                type: 'number',
                description: '每个分块中的字符数。默认为 1000。',
                default: 1000,
                optional: true
            },
            {
                label: '块重叠',
                name: 'chunkOverlap',
                type: 'number',
                description: '分块之间重叠的字符数。默认为 200。',
                default: 200,
                optional: true
            },
            {
                label: '自定义分隔符',
                name: 'separators',
                type: 'string',
                rows: 4,
                description: '自定义分隔符数组，用于确定何时分割文本，将覆盖默认分隔符',
                placeholder: `["|", "##", ">", "-"]`,
                additionalParams: true,
                optional: true
            }
        ]
    }

    async init(nodeData: INodeData): Promise<any> {
        const chunkSize = nodeData.inputs?.chunkSize as string
        const chunkOverlap = nodeData.inputs?.chunkOverlap as string
        const separators = nodeData.inputs?.separators

        const obj = {} as RecursiveCharacterTextSplitterParams

        if (chunkSize) obj.chunkSize = parseInt(chunkSize, 10)
        if (chunkOverlap) obj.chunkOverlap = parseInt(chunkOverlap, 10)
        if (separators) {
            try {
                obj.separators = typeof separators === 'object' ? separators : JSON.parse(separators)
            } catch (e) {
                throw new Error(e)
            }
        }

        const splitter = new RecursiveCharacterTextSplitter(obj)

        return splitter
    }
}

module.exports = { nodeClass: RecursiveCharacterTextSplitter_TextSplitters }
