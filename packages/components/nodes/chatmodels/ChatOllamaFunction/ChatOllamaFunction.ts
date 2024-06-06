import { HumanMessage, AIMessage, BaseMessage, AIMessageChunk, ChatMessage } from '@langchain/core/messages'
import { ChatResult } from '@langchain/core/outputs'
import { SimpleChatModel, BaseChatModel, BaseChatModelParams } from '@langchain/core/language_models/chat_models'
import { SystemMessagePromptTemplate } from '@langchain/core/prompts'
import { BaseCache } from '@langchain/core/caches'
import { type StructuredToolInterface } from '@langchain/core/tools'
import type { BaseFunctionCallOptions, BaseLanguageModelInput } from '@langchain/core/language_models/base'
import { convertToOpenAIFunction } from '@langchain/core/utils/function_calling'
import { RunnableInterface } from '@langchain/core/runnables'
import { ICommonObject, INode, INodeData, INodeParams } from '../../../src/Interface'
import { getBaseClasses } from '../../../src/utils'
import type { BaseLanguageModelCallOptions } from '@langchain/core/language_models/base'
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import type { StringWithAutocomplete } from '@langchain/core/utils/types'
import { createOllamaChatStream, createOllamaGenerateStream, type OllamaInput, type OllamaMessage } from './utils'

const DEFAULT_TOOL_SYSTEM_TEMPLATE = `You have access to the following tools:
{tools}
You must always select one of the above tools and respond with only a JSON object matching the following schema:
{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}`

class ChatOllamaFunction_ChatModels implements INode {
    label: string
    name: string
    version: number
    type: string
    icon: string
    category: string
    description: string
    baseClasses: string[]
    credential: INodeParams
    badge?: string
    inputs: INodeParams[]

    constructor() {
        this.label = 'ChatOllama 函数'
        this.name = 'chatOllamaFunction'
        this.version = 1.0
        this.type = 'ChatOllamaFunction'
        this.icon = 'Ollama.svg'
        this.category = '聊天模型'
        this.description = '在 Ollama 上运行开源函数调用兼容的 LLM'
        this.baseClasses = [this.type, ...getBaseClasses(OllamaFunctions)]
        this.badge = 'NEW'
        this.inputs = [
            {
                label: '缓存',
                name: 'cache',
                type: 'BaseCache',
                optional: true
            },
            {
                label: 'Base URL',
                name: 'baseUrl',
                type: 'string',
                default: 'http://localhost:11434'
            },
            {
                label: '模型名称',
                name: 'modelName',
                type: 'string',
                description: '仅与类似 mistral 的函数调用模型兼容',
                placeholder: 'mistral'
            },
            {
                label: '温度',
                name: 'temperature',
                type: 'number',
                description:
                    '模型的温度 。提高温度会使模型的回答更有创造性。(默认值:0.8)。更多细节请参考 <a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 0.1,
                default: 0.9,
                optional: true
            },
            {
                label: '工具系统提示',
                name: 'toolSystemPromptTemplate',
                type: 'string',
                rows: 4,
                description: `在底层实现中，Ollama 的 JSON 模式用于将输出限制为 JSON 格式。输出的 JSON 将包含两个键：tool 和 tool_input 字段。然后我们解析该 JSON 并执行工具。由于不同的模型具有不同的优势，因此可能有必要传递自己的系统提示。`,
                warning: `提示必须始终包含工具名称和指令，以便能够响应一个带有工具名称和工具输入字段的JSON对象。`,
                default: DEFAULT_TOOL_SYSTEM_TEMPLATE,
                placeholder: DEFAULT_TOOL_SYSTEM_TEMPLATE,
                additionalParams: true,
                optional: true
            },
            {
                label: 'Top P',
                name: 'topP',
                type: 'number',
                description:
                    '它与 top-k 一起使用。更高的值（例如 0.95）将导致生成更多样化的文本，而更低的值（例如 0.5）将生成更专注于保守的文本。（默认值：0.9）。有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 0.1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Top K',
                name: 'topK',
                type: 'number',
                description:
                    '减少产生无意义内容的概率。较高的值(如100)将给出更多样化的答案，而较低的值(如10)将更保守。(默认值:40) 有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Mirostat',
                name: 'mirostat',
                type: 'number',
                description:
                    '启用 Mirostat 采样以控制困惑度。(默认值:0,0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)。有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Mirostat ETA',
                name: 'mirostatEta',
                type: 'number',
                description:
                    '影响算法对生成文本的反馈的响应速度。较低的学习率将导致较慢的调整，而较高的学习率将使算法更具响应性。(默认:0.1) 有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 0.1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Mirostat TAU',
                name: 'mirostatTau',
                type: 'number',
                description:
                    '控制输出的一致性和多样性之间的平衡。值越低，文本越集中、越连贯。(默认:5.0) 有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 0.1,
                optional: true,
                additionalParams: true
            },
            {
                label: '上下文窗口大小',
                name: 'numCtx',
                type: 'number',
                description:
                    '设置用于生成下一个标记的上下文窗口的大小。(默认值:2048) 有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'GQA 组的数量',
                name: 'numGqa',
                type: 'number',
                description:
                    'transformer 层中 GQA 组的数量。对于某些型号是必需的，例如llama2:70b是8。有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 1,
                optional: true,
                additionalParams: true
            },
            {
                label: 'GPU 数量',
                name: 'numGpu',
                type: 'number',
                description:
                    '要发送到GPU的层数。在macOS上，默认值为1表示启用metal支持，0表示禁用。有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 1,
                optional: true,
                additionalParams: true
            },
            {
                label: '线程数',
                name: 'numThread',
                type: 'number',
                description:
                    '设置计算期间使用的线程数。默认情况下，Ollama将检测此参数以获得最佳性能。建议将此值设置为系统拥有的物理CPU内核数量(而不是逻辑内核数量)。有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 1,
                optional: true,
                additionalParams: true
            },
            {
                label: '重复最后 N 个',
                name: 'repeatLastN',
                type: 'number',
                description:
                    '设置模型向后看的距离，以防止重复。(默认值:64,0 = disabled， -1 = num_ctx)。有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 1,
                optional: true,
                additionalParams: true
            },
            {
                label: '惩罚重复',
                name: 'repeatPenalty',
                type: 'number',
                description:
                    '设定惩罚重复的力度。较高的值(如1.5)会更严厉地惩罚重复，而较低的值(如0.9)会更宽松。(默认值:1.1)。有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 0.1,
                optional: true,
                additionalParams: true
            },
            {
                label: '停止序列',
                name: 'stop',
                type: 'string',
                rows: 4,
                placeholder: 'AI assistant:',
                description:
                    '设置要使用的停止序列。使用逗号分隔不同的序列。 有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                optional: true,
                additionalParams: true
            },
            {
                label: '无尾采样',
                name: 'tfsZ',
                type: 'number',
                description:
                    '无尾采样用于减少输出中不太可能的标记的影响。较高的值(例如2.0)会更少影响，而1.0则禁用此设置。(默认:1)。有关更多详细信息，请参阅<a target="_blank" href="https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values">文档</a>',
                step: 0.1,
                optional: true,
                additionalParams: true
            }
        ]
    }

    async init(nodeData: INodeData): Promise<any> {
        const temperature = nodeData.inputs?.temperature as string
        const baseUrl = nodeData.inputs?.baseUrl as string
        const modelName = nodeData.inputs?.modelName as string
        const topP = nodeData.inputs?.topP as string
        const topK = nodeData.inputs?.topK as string
        const mirostat = nodeData.inputs?.mirostat as string
        const mirostatEta = nodeData.inputs?.mirostatEta as string
        const mirostatTau = nodeData.inputs?.mirostatTau as string
        const numCtx = nodeData.inputs?.numCtx as string
        const numGqa = nodeData.inputs?.numGqa as string
        const numGpu = nodeData.inputs?.numGpu as string
        const numThread = nodeData.inputs?.numThread as string
        const repeatLastN = nodeData.inputs?.repeatLastN as string
        const repeatPenalty = nodeData.inputs?.repeatPenalty as string
        const stop = nodeData.inputs?.stop as string
        const tfsZ = nodeData.inputs?.tfsZ as string
        const toolSystemPromptTemplate = nodeData.inputs?.toolSystemPromptTemplate as string

        const cache = nodeData.inputs?.cache as BaseCache

        const obj: OllamaFunctionsInput = {
            baseUrl,
            temperature: parseFloat(temperature),
            model: modelName,
            toolSystemPromptTemplate: toolSystemPromptTemplate ? toolSystemPromptTemplate : DEFAULT_TOOL_SYSTEM_TEMPLATE
        }

        if (topP) obj.topP = parseFloat(topP)
        if (topK) obj.topK = parseFloat(topK)
        if (mirostat) obj.mirostat = parseFloat(mirostat)
        if (mirostatEta) obj.mirostatEta = parseFloat(mirostatEta)
        if (mirostatTau) obj.mirostatTau = parseFloat(mirostatTau)
        if (numCtx) obj.numCtx = parseFloat(numCtx)
        if (numGqa) obj.numGqa = parseFloat(numGqa)
        if (numGpu) obj.numGpu = parseFloat(numGpu)
        if (numThread) obj.numThread = parseFloat(numThread)
        if (repeatLastN) obj.repeatLastN = parseFloat(repeatLastN)
        if (repeatPenalty) obj.repeatPenalty = parseFloat(repeatPenalty)
        if (tfsZ) obj.tfsZ = parseFloat(tfsZ)
        if (stop) {
            const stopSequences = stop.split(',')
            obj.stop = stopSequences
        }
        if (cache) obj.cache = cache

        const model = new OllamaFunctions(obj)
        return model
    }
}

interface ChatOllamaFunctionsCallOptions extends BaseFunctionCallOptions {}

type OllamaFunctionsInput = Partial<ChatOllamaInput> &
    BaseChatModelParams & {
        llm?: OllamaChat
        toolSystemPromptTemplate?: string
    }

class OllamaFunctions extends BaseChatModel<ChatOllamaFunctionsCallOptions> {
    llm: OllamaChat

    fields?: OllamaFunctionsInput

    toolSystemPromptTemplate: string = DEFAULT_TOOL_SYSTEM_TEMPLATE

    protected defaultResponseFunction = {
        name: '__conversational_response',
        description: 'Respond conversationally if no other tools should be called for a given query.',
        parameters: {
            type: 'object',
            properties: {
                response: {
                    type: 'string',
                    description: 'Conversational response to the user.'
                }
            },
            required: ['response']
        }
    }

    static lc_name(): string {
        return 'OllamaFunctions'
    }

    constructor(fields?: OllamaFunctionsInput) {
        super(fields ?? {})
        this.fields = fields
        this.llm = fields?.llm ?? new OllamaChat({ ...fields, format: 'json' })
        this.toolSystemPromptTemplate = fields?.toolSystemPromptTemplate ?? this.toolSystemPromptTemplate
    }

    invocationParams() {
        return this.llm.invocationParams()
    }

    /** @ignore */
    _identifyingParams() {
        return this.llm._identifyingParams()
    }

    async _generate(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun | undefined
    ): Promise<ChatResult> {
        let functions = options.functions ?? []
        if (options.function_call !== undefined) {
            functions = functions.filter((fn) => fn.name === options.function_call?.name)
            if (!functions.length) {
                throw new Error(`If "function_call" is specified, you must also pass a matching function in "functions".`)
            }
        } else if (functions.length === 0) {
            functions.push(this.defaultResponseFunction)
        }
        const systemPromptTemplate = SystemMessagePromptTemplate.fromTemplate(this.toolSystemPromptTemplate)
        const systemMessage = await systemPromptTemplate.format({
            tools: JSON.stringify(functions, null, 2)
        })

        let generatedMessages = [systemMessage, ...messages]
        let isToolResponse = false
        if (
            messages.length > 3 &&
            messages[messages.length - 1]._getType() === 'tool' &&
            functions.length &&
            messages[messages.length - 1].additional_kwargs?.name === functions[0].name
        ) {
            const lastToolQuestion = messages[messages.length - 3].content
            const lastToolResp = messages.pop()?.content
            // Pop the message again to get rid of tool call message
            messages.pop()?.content
            const humanMessage = new HumanMessage({
                content: `Given user question: ${lastToolQuestion} and answer: ${lastToolResp}\n\nWrite a natural language response`
            })
            generatedMessages = [...messages, humanMessage]
            isToolResponse = true
            this.llm = new OllamaChat({ ...this.fields })
        }
        const chatResult = await this.llm._generate(generatedMessages, options, runManager)
        const chatGenerationContent = chatResult.generations[0].message.content

        if (typeof chatGenerationContent !== 'string') {
            throw new Error('OllamaFunctions does not support non-string output.')
        }

        if (isToolResponse) {
            return {
                generations: [
                    {
                        message: new AIMessage({
                            content: chatGenerationContent
                        }),
                        text: chatGenerationContent
                    }
                ]
            }
        }

        let parsedChatResult
        try {
            parsedChatResult = JSON.parse(chatGenerationContent)
        } catch (e) {
            throw new Error(`"${this.llm.model}" did not respond with valid JSON. Please try again.`)
        }

        const calledToolName = parsedChatResult.tool
        const calledToolArguments = parsedChatResult.tool_input
        const calledTool = functions.find((fn) => fn.name === calledToolName)
        if (calledTool === undefined) {
            throw new Error(`Failed to parse a function call from ${this.llm.model} output: ${chatGenerationContent}`)
        }

        if (calledTool.name === this.defaultResponseFunction.name) {
            return {
                generations: [
                    {
                        message: new AIMessage({
                            content: calledToolArguments.response
                        }),
                        text: calledToolArguments.response
                    }
                ]
            }
        }

        const responseMessageWithFunctions = new AIMessage({
            content: '',
            tool_calls: [
                {
                    name: calledToolName,
                    args: calledToolArguments || {}
                }
            ],
            invalid_tool_calls: [],
            additional_kwargs: {
                function_call: {
                    name: calledToolName,
                    arguments: calledToolArguments ? JSON.stringify(calledToolArguments) : ''
                },
                tool_calls: [
                    {
                        id: Date.now().toString(),
                        type: 'function',
                        function: {
                            name: calledToolName,
                            arguments: calledToolArguments ? JSON.stringify(calledToolArguments) : ''
                        }
                    }
                ]
            }
        })

        return {
            generations: [{ message: responseMessageWithFunctions, text: '' }]
        }
    }

    override bindTools(
        tools: StructuredToolInterface[],
        kwargs?: Partial<ICommonObject>
    ): RunnableInterface<BaseLanguageModelInput, AIMessageChunk, ICommonObject> {
        return this.bind({
            functions: tools.map((tool) => convertToOpenAIFunction(tool)),
            ...kwargs
        } as Partial<ICommonObject>)
    }

    _llmType(): string {
        return 'ollama_functions'
    }

    /** @ignore */
    _combineLLMOutput() {
        return []
    }
}

export interface ChatOllamaInput extends OllamaInput {}

interface ChatOllamaCallOptions extends BaseLanguageModelCallOptions {}

class OllamaChat extends SimpleChatModel<ChatOllamaCallOptions> implements ChatOllamaInput {
    static lc_name() {
        return 'ChatOllama'
    }

    lc_serializable = true

    model = 'llama2'

    baseUrl = 'http://localhost:11434'

    keepAlive = '5m'

    embeddingOnly?: boolean

    f16KV?: boolean

    frequencyPenalty?: number

    headers?: Record<string, string>

    logitsAll?: boolean

    lowVram?: boolean

    mainGpu?: number

    mirostat?: number

    mirostatEta?: number

    mirostatTau?: number

    numBatch?: number

    numCtx?: number

    numGpu?: number

    numGqa?: number

    numKeep?: number

    numPredict?: number

    numThread?: number

    penalizeNewline?: boolean

    presencePenalty?: number

    repeatLastN?: number

    repeatPenalty?: number

    ropeFrequencyBase?: number

    ropeFrequencyScale?: number

    temperature?: number

    stop?: string[]

    tfsZ?: number

    topK?: number

    topP?: number

    typicalP?: number

    useMLock?: boolean

    useMMap?: boolean

    vocabOnly?: boolean

    format?: StringWithAutocomplete<'json'>

    constructor(fields: OllamaInput & BaseChatModelParams) {
        super(fields)
        this.model = fields.model ?? this.model
        this.baseUrl = fields.baseUrl?.endsWith('/') ? fields.baseUrl.slice(0, -1) : fields.baseUrl ?? this.baseUrl
        this.keepAlive = fields.keepAlive ?? this.keepAlive
        this.embeddingOnly = fields.embeddingOnly
        this.f16KV = fields.f16KV
        this.frequencyPenalty = fields.frequencyPenalty
        this.headers = fields.headers
        this.logitsAll = fields.logitsAll
        this.lowVram = fields.lowVram
        this.mainGpu = fields.mainGpu
        this.mirostat = fields.mirostat
        this.mirostatEta = fields.mirostatEta
        this.mirostatTau = fields.mirostatTau
        this.numBatch = fields.numBatch
        this.numCtx = fields.numCtx
        this.numGpu = fields.numGpu
        this.numGqa = fields.numGqa
        this.numKeep = fields.numKeep
        this.numPredict = fields.numPredict
        this.numThread = fields.numThread
        this.penalizeNewline = fields.penalizeNewline
        this.presencePenalty = fields.presencePenalty
        this.repeatLastN = fields.repeatLastN
        this.repeatPenalty = fields.repeatPenalty
        this.ropeFrequencyBase = fields.ropeFrequencyBase
        this.ropeFrequencyScale = fields.ropeFrequencyScale
        this.temperature = fields.temperature
        this.stop = fields.stop
        this.tfsZ = fields.tfsZ
        this.topK = fields.topK
        this.topP = fields.topP
        this.typicalP = fields.typicalP
        this.useMLock = fields.useMLock
        this.useMMap = fields.useMMap
        this.vocabOnly = fields.vocabOnly
        this.format = fields.format
    }

    _llmType() {
        return 'ollama'
    }

    /**
     * A method that returns the parameters for an Ollama API call. It
     * includes model and options parameters.
     * @param options Optional parsed call options.
     * @returns An object containing the parameters for an Ollama API call.
     */
    invocationParams(options?: this['ParsedCallOptions']) {
        return {
            model: this.model,
            format: this.format,
            keep_alive: this.keepAlive,
            options: {
                embedding_only: this.embeddingOnly,
                f16_kv: this.f16KV,
                frequency_penalty: this.frequencyPenalty,
                logits_all: this.logitsAll,
                low_vram: this.lowVram,
                main_gpu: this.mainGpu,
                mirostat: this.mirostat,
                mirostat_eta: this.mirostatEta,
                mirostat_tau: this.mirostatTau,
                num_batch: this.numBatch,
                num_ctx: this.numCtx,
                num_gpu: this.numGpu,
                num_gqa: this.numGqa,
                num_keep: this.numKeep,
                num_predict: this.numPredict,
                num_thread: this.numThread,
                penalize_newline: this.penalizeNewline,
                presence_penalty: this.presencePenalty,
                repeat_last_n: this.repeatLastN,
                repeat_penalty: this.repeatPenalty,
                rope_frequency_base: this.ropeFrequencyBase,
                rope_frequency_scale: this.ropeFrequencyScale,
                temperature: this.temperature,
                stop: options?.stop ?? this.stop,
                tfs_z: this.tfsZ,
                top_k: this.topK,
                top_p: this.topP,
                typical_p: this.typicalP,
                use_mlock: this.useMLock,
                use_mmap: this.useMMap,
                vocab_only: this.vocabOnly
            }
        }
    }

    _combineLLMOutput() {
        return {}
    }

    /** @deprecated */
    async *_streamResponseChunksLegacy(
        input: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
        const stream = createOllamaGenerateStream(
            this.baseUrl,
            {
                ...this.invocationParams(options),
                prompt: this._formatMessagesAsPrompt(input)
            },
            {
                ...options,
                headers: this.headers
            }
        )
        for await (const chunk of stream) {
            if (!chunk.done) {
                yield new ChatGenerationChunk({
                    text: chunk.response,
                    message: new AIMessageChunk({ content: chunk.response })
                })
                await runManager?.handleLLMNewToken(chunk.response ?? '')
            } else {
                yield new ChatGenerationChunk({
                    text: '',
                    message: new AIMessageChunk({ content: '' }),
                    generationInfo: {
                        model: chunk.model,
                        total_duration: chunk.total_duration,
                        load_duration: chunk.load_duration,
                        prompt_eval_count: chunk.prompt_eval_count,
                        prompt_eval_duration: chunk.prompt_eval_duration,
                        eval_count: chunk.eval_count,
                        eval_duration: chunk.eval_duration
                    }
                })
            }
        }
    }

    async *_streamResponseChunks(
        input: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
        try {
            const stream = await this.caller.call(async () =>
                createOllamaChatStream(
                    this.baseUrl,
                    {
                        ...this.invocationParams(options),
                        messages: this._convertMessagesToOllamaMessages(input)
                    },
                    {
                        ...options,
                        headers: this.headers
                    }
                )
            )
            for await (const chunk of stream) {
                if (!chunk.done) {
                    yield new ChatGenerationChunk({
                        text: chunk.message.content,
                        message: new AIMessageChunk({ content: chunk.message.content })
                    })
                    await runManager?.handleLLMNewToken(chunk.message.content ?? '')
                } else {
                    yield new ChatGenerationChunk({
                        text: '',
                        message: new AIMessageChunk({ content: '' }),
                        generationInfo: {
                            model: chunk.model,
                            total_duration: chunk.total_duration,
                            load_duration: chunk.load_duration,
                            prompt_eval_count: chunk.prompt_eval_count,
                            prompt_eval_duration: chunk.prompt_eval_duration,
                            eval_count: chunk.eval_count,
                            eval_duration: chunk.eval_duration
                        }
                    })
                }
            }
        } catch (e: any) {
            if (e.response?.status === 404) {
                console.warn(
                    '[WARNING]: It seems you are using a legacy version of Ollama. Please upgrade to a newer version for better chat support.'
                )
                yield* this._streamResponseChunksLegacy(input, options, runManager)
            } else {
                throw e
            }
        }
    }

    protected _convertMessagesToOllamaMessages(messages: BaseMessage[]): OllamaMessage[] {
        return messages.map((message) => {
            let role
            if (message._getType() === 'human') {
                role = 'user'
            } else if (message._getType() === 'ai' || message._getType() === 'tool') {
                role = 'assistant'
            } else if (message._getType() === 'system') {
                role = 'system'
            } else {
                throw new Error(`Unsupported message type for Ollama: ${message._getType()}`)
            }
            let content = ''
            const images = []
            if (typeof message.content === 'string') {
                content = message.content
            } else {
                for (const contentPart of message.content) {
                    if (contentPart.type === 'text') {
                        content = `${content}\n${contentPart.text}`
                    } else if (contentPart.type === 'image_url' && typeof contentPart.image_url === 'string') {
                        const imageUrlComponents = contentPart.image_url.split(',')
                        // Support both data:image/jpeg;base64,<image> format as well
                        images.push(imageUrlComponents[1] ?? imageUrlComponents[0])
                    } else {
                        throw new Error(
                            `Unsupported message content type. Must either have type "text" or type "image_url" with a string "image_url" field.`
                        )
                    }
                }
            }
            return {
                role,
                content,
                images
            }
        })
    }

    /** @deprecated */
    protected _formatMessagesAsPrompt(messages: BaseMessage[]): string {
        const formattedMessages = messages
            .map((message) => {
                let messageText
                if (message._getType() === 'human') {
                    messageText = `[INST] ${message.content} [/INST]`
                } else if (message._getType() === 'ai') {
                    messageText = message.content
                } else if (message._getType() === 'system') {
                    messageText = `<<SYS>> ${message.content} <</SYS>>`
                } else if (ChatMessage.isInstance(message)) {
                    messageText = `\n\n${message.role[0].toUpperCase()}${message.role.slice(1)}: ${message.content}`
                } else {
                    console.warn(`Unsupported message type passed to Ollama: "${message._getType()}"`)
                    messageText = ''
                }
                return messageText
            })
            .join('\n')
        return formattedMessages
    }

    /** @ignore */
    async _call(messages: BaseMessage[], options: this['ParsedCallOptions'], runManager?: CallbackManagerForLLMRun): Promise<string> {
        const chunks = []
        for await (const chunk of this._streamResponseChunks(messages, options, runManager)) {
            chunks.push(chunk.message.content)
        }
        return chunks.join('')
    }
}

module.exports = { nodeClass: ChatOllamaFunction_ChatModels }
