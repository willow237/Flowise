import { ICommonObject, IDatabaseEntity, INode, INodeData, INodeOptionsValue, INodeOutputsValue, INodeParams } from '../../../src/Interface'
import { DataSource } from 'typeorm'
import { Document } from '@langchain/core/documents'
import { handleEscapeCharacters } from '../../../src'

class DocStore_DocumentLoaders implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    inputs: INodeParams[]
    outputs: INodeOutputsValue[]
    badge: string

    constructor() {
        this.label = '文件存储'
        this.name = 'documentStore'
        this.version = 1.0
        this.type = 'Document'
        this.icon = 'dstore.svg'
        this.badge = 'NEW'
        this.category = '文档加载器'
        this.description = `从预先配置的文件存储区加载数据`
        this.baseClasses = [this.type]
        this.inputs = [
            {
                label: '选择存储',
                name: 'selectedStore',
                type: 'asyncOptions',
                loadMethod: 'listStores'
            }
        ]
        this.outputs = [
            {
                label: '文档',
                name: 'document',
                description: '包含元数据和页面内容的文档对象数组',
                baseClasses: [...this.baseClasses, 'json']
            },
            {
                label: '文本',
                name: 'text',
                description: '从文件的页面内容中提取的并集字符串',
                baseClasses: ['string', 'json']
            }
        ]
    }

    //@ts-ignore
    loadMethods = {
        async listStores(_: INodeData, options: ICommonObject): Promise<INodeOptionsValue[]> {
            const returnData: INodeOptionsValue[] = []

            const appDataSource = options.appDataSource as DataSource
            const databaseEntities = options.databaseEntities as IDatabaseEntity

            if (appDataSource === undefined || !appDataSource) {
                return returnData
            }

            const stores = await appDataSource.getRepository(databaseEntities['DocumentStore']).find()
            for (const store of stores) {
                if (store.status === 'SYNC') {
                    const obj = {
                        name: store.id,
                        label: store.name,
                        description: store.description
                    }
                    returnData.push(obj)
                }
            }
            return returnData
        }
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const selectedStore = nodeData.inputs?.selectedStore as string
        const appDataSource = options.appDataSource as DataSource
        const databaseEntities = options.databaseEntities as IDatabaseEntity
        const chunks = await appDataSource
            .getRepository(databaseEntities['DocumentStoreFileChunk'])
            .find({ where: { storeId: selectedStore } })
        const output = nodeData.outputs?.output as string

        const finalDocs = []
        for (const chunk of chunks) {
            finalDocs.push(new Document({ pageContent: chunk.pageContent, metadata: JSON.parse(chunk.metadata) }))
        }

        if (output === 'document') {
            return finalDocs
        } else {
            let finaltext = ''
            for (const doc of finalDocs) {
                finaltext += `${doc.pageContent}\n`
            }
            return handleEscapeCharacters(finaltext, false)
        }
    }
}

module.exports = { nodeClass: DocStore_DocumentLoaders }
