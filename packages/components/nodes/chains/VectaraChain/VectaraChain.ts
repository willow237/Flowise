import fetch from 'node-fetch'
import { Document } from '@langchain/core/documents'
import { VectaraStore } from '@langchain/community/vectorstores/vectara'
import { VectorDBQAChain } from 'langchain/chains'
import { INode, INodeData, INodeParams } from '../../../src/Interface'
import { getBaseClasses } from '../../../src/utils'
import { checkInputs, Moderation } from '../../moderation/Moderation'
import { formatResponse } from '../../outputparsers/OutputParserHelpers'

// functionality based on https://github.com/vectara/vectara-answer
const reorderCitations = (unorderedSummary: string) => {
    const allCitations = unorderedSummary.match(/\[\d+\]/g) || []

    const uniqueCitations = [...new Set(allCitations)]
    const citationToReplacement: { [key: string]: string } = {}
    uniqueCitations.forEach((citation, index) => {
        citationToReplacement[citation] = `[${index + 1}]`
    })

    return unorderedSummary.replace(/\[\d+\]/g, (match) => citationToReplacement[match])
}
const applyCitationOrder = (searchResults: any[], unorderedSummary: string) => {
    const orderedSearchResults: any[] = []
    const allCitations = unorderedSummary.match(/\[\d+\]/g) || []

    const addedIndices = new Set<number>()
    for (let i = 0; i < allCitations.length; i++) {
        const citation = allCitations[i]
        const index = Number(citation.slice(1, citation.length - 1)) - 1

        if (addedIndices.has(index)) continue
        orderedSearchResults.push(searchResults[index])
        addedIndices.add(index)
    }

    return orderedSearchResults
}

class VectaraChain_Chains implements INode {
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
        this.label = 'Vectara QA 对话链'
        this.name = 'vectaraQAChain'
        this.version = 2.0
        this.type = 'VectaraQAChain'
        this.icon = 'vectara.png'
        this.category = '对话链'
        this.description = '用于 Vectara 的 QA 对话链'
        this.baseClasses = [this.type, ...getBaseClasses(VectorDBQAChain)]
        this.inputs = [
            {
                label: 'Vectara 存储',
                name: 'vectaraStore',
                type: 'VectorStore'
            },
            {
                label: '摘要 Prompt 名称',
                name: 'summarizerPromptName',
                description:
                    '总结从 Vectara 获取的结果。 请阅读<a target="_blank" href="https://docs.vectara.com/docs/learn/grounded-generation/select-a-summarizer">更多</a>',
                type: 'options',
                options: [
                    {
                        label: 'vectara-summary-ext-v1.2.0 (gpt-3.5-turbo)',
                        name: 'vectara-summary-ext-v1.2.0',
                        description: '所有 Vectara 用户均可使用的基础摘要器'
                    },
                    {
                        label: 'vectara-experimental-summary-ext-2023-10-23-small (gpt-3.5-turbo)',
                        name: 'vectara-experimental-summary-ext-2023-10-23-small',
                        description: `测试版，面向 Growth 和 <a target="_blank" href="https://vectara.com/pricing/">Scale</a> Vectara 用户开放`
                    },
                    {
                        label: 'vectara-summary-ext-v1.3.0 (gpt-4.0)',
                        name: 'vectara-summary-ext-v1.3.0',
                        description: '仅适用于 <a target="_blank" href="https://vectara.com/pricing/">Scale</a> Vectara 用户'
                    },
                    {
                        label: 'vectara-experimental-summary-ext-2023-10-23-med (gpt-4.0)',
                        name: 'vectara-experimental-summary-ext-2023-10-23-med',
                        description: `测试版，仅面向 <a target="_blank" href="https://vectara.com/pricing/">Scale</a> Vectara 用户开放`
                    }
                ],
                default: 'vectara-summary-ext-v1.2.0'
            },
            {
                label: '响应语言',
                name: 'responseLang',
                description:
                    '以特定语言返回响应。如果未选择，Vectara 将自动检测语言。阅读<a target="_blank" href="https://docs.vectara.com/docs/learn/grounded-generation/grounded-generation-response-languages">更多</a>',
                type: 'options',
                options: [
                    {
                        label: '英语',
                        name: 'eng'
                    },
                    {
                        label: '德语',
                        name: 'deu'
                    },
                    {
                        label: '法语',
                        name: 'fra'
                    },
                    {
                        label: '中文',
                        name: 'zho'
                    },
                    {
                        label: '韩语',
                        name: 'kor'
                    },
                    {
                        label: '阿拉伯语',
                        name: 'ara'
                    },
                    {
                        label: '俄语',
                        name: 'rus'
                    },
                    {
                        label: '泰语',
                        name: 'tha'
                    },
                    {
                        label: '荷兰语',
                        name: 'nld'
                    },
                    {
                        label: '意大利语',
                        name: 'ita'
                    },
                    {
                        label: '葡萄牙语',
                        name: 'por'
                    },
                    {
                        label: '西班牙语',
                        name: 'spa'
                    },
                    {
                        label: '日语',
                        name: 'jpn'
                    },
                    {
                        label: '波兰语',
                        name: 'pol'
                    },
                    {
                        label: '土耳其语',
                        name: 'tur'
                    },
                    {
                        label: '越南语',
                        name: 'vie'
                    },
                    {
                        label: '印尼语',
                        name: 'ind'
                    },
                    {
                        label: '捷克语',
                        name: 'ces'
                    },
                    {
                        label: '乌克兰语',
                        name: 'ukr'
                    },
                    {
                        label: '希腊语',
                        name: 'ell'
                    },
                    {
                        label: '希伯来语',
                        name: 'heb'
                    },
                    {
                        label: '波斯语',
                        name: 'fas'
                    },
                    {
                        label: '北印度语',
                        name: 'hin'
                    },
                    {
                        label: '乌尔都语',
                        name: 'urd'
                    },
                    {
                        label: '瑞典语',
                        name: 'swe'
                    },
                    {
                        label: '孟加拉语',
                        name: 'ben'
                    },
                    {
                        label: '马来语',
                        name: 'msa'
                    },
                    {
                        label: '罗马尼亚语',
                        name: 'ron'
                    }
                ],
                optional: true,
                default: 'eng'
            },
            {
                label: '最大汇总结果',
                name: 'maxSummarizedResults',
                description: '用于建立汇总响应的最大结果',
                type: 'number',
                default: 7
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

    async init(): Promise<any> {
        return null
    }

    async run(nodeData: INodeData, input: string): Promise<string | object> {
        const vectorStore = nodeData.inputs?.vectaraStore as VectaraStore
        const responseLang = (nodeData.inputs?.responseLang as string) ?? 'eng'
        const summarizerPromptName = nodeData.inputs?.summarizerPromptName as string
        const maxSummarizedResultsStr = nodeData.inputs?.maxSummarizedResults as string
        const maxSummarizedResults = maxSummarizedResultsStr ? parseInt(maxSummarizedResultsStr, 10) : 7

        const topK = (vectorStore as any)?.k ?? 10

        const headers = await vectorStore.getJsonHeader()
        const vectaraFilter = (vectorStore as any).vectaraFilter ?? {}
        const corpusId: number[] = (vectorStore as any).corpusId ?? []
        const customerId = (vectorStore as any).customerId ?? ''

        const corpusKeys = corpusId.map((corpusId) => ({
            customerId,
            corpusId,
            metadataFilter: vectaraFilter?.filter ?? '',
            lexicalInterpolationConfig: { lambda: vectaraFilter?.lambda ?? 0.025 }
        }))

        // Vectara reranker ID for MMR (https://docs.vectara.com/docs/api-reference/search-apis/reranking#maximal-marginal-relevance-mmr-reranker)
        const mmrRerankerId = 272725718
        const mmrEnabled = vectaraFilter?.mmrConfig?.enabled

        const moderations = nodeData.inputs?.inputModeration as Moderation[]
        if (moderations && moderations.length > 0) {
            try {
                // Use the output of the moderation chain as input for the Vectara chain
                input = await checkInputs(moderations, input)
            } catch (e) {
                await new Promise((resolve) => setTimeout(resolve, 500))
                //streamResponse(options.socketIO && options.socketIOClientId, e.message, options.socketIO, options.socketIOClientId)
                return formatResponse(e.message)
            }
        }

        const data = {
            query: [
                {
                    query: input,
                    start: 0,
                    numResults: mmrEnabled ? vectaraFilter?.mmrTopK : topK,
                    corpusKey: corpusKeys,
                    contextConfig: {
                        sentencesAfter: vectaraFilter?.contextConfig?.sentencesAfter ?? 2,
                        sentencesBefore: vectaraFilter?.contextConfig?.sentencesBefore ?? 2
                    },
                    ...(mmrEnabled
                        ? {
                              rerankingConfig: {
                                  rerankerId: mmrRerankerId,
                                  mmrConfig: {
                                      diversityBias: vectaraFilter?.mmrConfig.diversityBias
                                  }
                              }
                          }
                        : {}),
                    summary: [
                        {
                            summarizerPromptName,
                            responseLang,
                            maxSummarizedResults
                        }
                    ]
                }
            ]
        }

        try {
            const response = await fetch(`https://api.vectara.io/v1/query`, {
                method: 'POST',
                headers: headers?.headers,
                body: JSON.stringify(data)
            })

            if (response.status !== 200) {
                throw new Error(`Vectara API returned status code ${response.status}`)
            }

            const result = await response.json()
            const responses = result.responseSet[0].response
            const documents = result.responseSet[0].document
            let rawSummarizedText = ''

            // remove responses that are not in the topK (in case of MMR)
            // Note that this does not really matter functionally due to the reorder citations, but it is more efficient
            const maxResponses = mmrEnabled ? Math.min(responses.length, topK) : responses.length
            if (responses.length > maxResponses) {
                responses.splice(0, maxResponses)
            }

            // Add metadata to each text response given its corresponding document metadata
            for (let i = 0; i < responses.length; i += 1) {
                const responseMetadata = responses[i].metadata
                const documentMetadata = documents[responses[i].documentIndex].metadata
                const combinedMetadata: Record<string, unknown> = {}

                responseMetadata.forEach((item: { name: string; value: unknown }) => {
                    combinedMetadata[item.name] = item.value
                })

                documentMetadata.forEach((item: { name: string; value: unknown }) => {
                    combinedMetadata[item.name] = item.value
                })

                responses[i].metadata = combinedMetadata
            }

            // Create the summarization response
            const summaryStatus = result.responseSet[0].summary[0].status
            if (summaryStatus.length > 0 && summaryStatus[0].code === 'BAD_REQUEST') {
                throw new Error(
                    `BAD REQUEST: Too much text for the summarizer to summarize. Please try reducing the number of search results to summarize, or the context of each result by adjusting the 'summary_num_sentences', and 'summary_num_results' parameters respectively.`
                )
            }
            if (
                summaryStatus.length > 0 &&
                summaryStatus[0].code === 'NOT_FOUND' &&
                summaryStatus[0].statusDetail === 'Failed to retrieve summarizer.'
            ) {
                throw new Error(`BAD REQUEST: summarizer ${summarizerPromptName} is invalid for this account.`)
            }

            // Reorder citations in summary and create the list of returned source documents
            rawSummarizedText = result.responseSet[0].summary[0]?.text
            let summarizedText = reorderCitations(rawSummarizedText)
            let summaryResponses = applyCitationOrder(responses, rawSummarizedText)

            const sourceDocuments: Document[] = summaryResponses.map(
                (response: { text: string; metadata: Record<string, unknown>; score: number }) =>
                    new Document({
                        pageContent: response.text,
                        metadata: response.metadata
                    })
            )

            return { text: summarizedText, sourceDocuments: sourceDocuments }
        } catch (error) {
            throw new Error(error)
        }
    }
}

module.exports = { nodeClass: VectaraChain_Chains }
