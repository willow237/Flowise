import axios, { AxiosRequestConfig } from 'axios'
import { omit } from 'lodash'
import { Document } from '@langchain/core/documents'
import { TextSplitter } from 'langchain/text_splitter'
import { BaseDocumentLoader } from 'langchain/document_loaders/base'
import { ICommonObject, IDocument, INode, INodeData, INodeParams } from '../../../src/Interface'

class API_DocumentLoaders implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    inputs?: INodeParams[]

    constructor() {
        this.label = 'API 加载器'
        this.name = 'apiLoader'
        this.version = 1.0
        this.type = 'Document'
        this.icon = 'api.svg'
        this.category = '文档加载器'
        this.description = `从 API 加载数据`
        this.baseClasses = [this.type]
        this.inputs = [
            {
                label: '文本分割器',
                name: 'textSplitter',
                type: 'TextSplitter',
                optional: true
            },
            {
                label: '方法',
                name: 'method',
                type: 'options',
                options: [
                    {
                        label: 'GET',
                        name: 'GET'
                    },
                    {
                        label: 'POST',
                        name: 'POST'
                    }
                ]
            },
            {
                label: 'URL',
                name: 'url',
                type: 'string'
            },
            {
                label: 'Headers',
                name: 'headers',
                type: 'json',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Body',
                name: 'body',
                type: 'json',
                description: 'POST 请求的 JSON 主体。如果未指定，代理将尝试从 AIPlugin（如果提供）中自行查找。',
                additionalParams: true,
                optional: true
            },
            {
                label: '附加元数据',
                name: 'metadata',
                type: 'json',
                description: '在提取的文件中添加附加元数据',
                optional: true,
                additionalParams: true
            },
            {
                label: '省略元数据键',
                name: 'omitMetadataKeys',
                type: 'string',
                rows: 4,
                description:
                    '每个文档加载器都有一组从文档中提取的默认元数据键。您可以使用此字段省略某些默认元数据键。值应该是一个键的列表，用逗号分隔。使用 * 可省略所有元数据关键字，但在附加元数据字段中指定的关键字除外',
                placeholder: 'key1, key2, key3.nestedKey1',
                optional: true,
                additionalParams: true
            }
        ]
    }
    async init(nodeData: INodeData): Promise<any> {
        const headers = nodeData.inputs?.headers as string
        const url = nodeData.inputs?.url as string
        const body = nodeData.inputs?.body as string
        const method = nodeData.inputs?.method as string
        const textSplitter = nodeData.inputs?.textSplitter as TextSplitter
        const metadata = nodeData.inputs?.metadata
        const _omitMetadataKeys = nodeData.inputs?.omitMetadataKeys as string

        let omitMetadataKeys: string[] = []
        if (_omitMetadataKeys) {
            omitMetadataKeys = _omitMetadataKeys.split(',').map((key) => key.trim())
        }

        const options: ApiLoaderParams = {
            url,
            method
        }

        if (headers) {
            const parsedHeaders = typeof headers === 'object' ? headers : JSON.parse(headers)
            options.headers = parsedHeaders
        }

        if (body) {
            const parsedBody = typeof body === 'object' ? body : JSON.parse(body)
            options.body = parsedBody
        }

        const loader = new ApiLoader(options)

        let docs: IDocument[] = []

        if (textSplitter) {
            docs = await loader.loadAndSplit(textSplitter)
        } else {
            docs = await loader.load()
        }

        if (metadata) {
            const parsedMetadata = typeof metadata === 'object' ? metadata : JSON.parse(metadata)
            docs = docs.map((doc) => ({
                ...doc,
                metadata:
                    _omitMetadataKeys === '*'
                        ? {
                              ...parsedMetadata
                          }
                        : omit(
                              {
                                  ...doc.metadata,
                                  ...parsedMetadata
                              },
                              omitMetadataKeys
                          )
            }))
        } else {
            docs = docs.map((doc) => ({
                ...doc,
                metadata:
                    _omitMetadataKeys === '*'
                        ? {}
                        : omit(
                              {
                                  ...doc.metadata
                              },
                              omitMetadataKeys
                          )
            }))
        }

        return docs
    }
}

interface ApiLoaderParams {
    url: string
    method: string
    headers?: ICommonObject
    body?: ICommonObject
}

class ApiLoader extends BaseDocumentLoader {
    public readonly url: string

    public readonly headers?: ICommonObject

    public readonly body?: ICommonObject

    public readonly method: string

    constructor({ url, headers, body, method }: ApiLoaderParams) {
        super()
        this.url = url
        this.headers = headers
        this.body = body
        this.method = method
    }

    public async load(): Promise<IDocument[]> {
        if (this.method === 'POST') {
            return this.executePostRequest(this.url, this.headers, this.body)
        } else {
            return this.executeGetRequest(this.url, this.headers)
        }
    }

    protected async executeGetRequest(url: string, headers?: ICommonObject): Promise<IDocument[]> {
        try {
            const config: AxiosRequestConfig = {}
            if (headers) {
                config.headers = headers
            }
            const response = await axios.get(url, config)
            const responseJsonString = JSON.stringify(response.data, null, 2)
            const doc = new Document({
                pageContent: responseJsonString,
                metadata: {
                    url
                }
            })
            return [doc]
        } catch (error) {
            throw new Error(`Failed to fetch ${url}: ${error}`)
        }
    }

    protected async executePostRequest(url: string, headers?: ICommonObject, body?: ICommonObject): Promise<IDocument[]> {
        try {
            const config: AxiosRequestConfig = {}
            if (headers) {
                config.headers = headers
            }
            const response = await axios.post(url, body ?? {}, config)
            const responseJsonString = JSON.stringify(response.data, null, 2)
            const doc = new Document({
                pageContent: responseJsonString,
                metadata: {
                    url
                }
            })
            return [doc]
        } catch (error) {
            throw new Error(`Failed to post ${url}: ${error}`)
        }
    }
}

module.exports = {
    nodeClass: API_DocumentLoaders
}
