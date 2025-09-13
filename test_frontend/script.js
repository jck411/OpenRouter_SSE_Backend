class ChatApp {
    constructor() {
        this.baseUrl = 'http://localhost:8000';
        this.currentModel = '';
        this.messages = [];
        this.isStreaming = false;
        this.allModels = [];
        this.modelCapabilities = {}; // Store supported parameters for each model
        this.currentUsageData = null; // Store usage data for most recent message (for compatibility)
        this.usageByMessageId = new Map(); // Store usage data per assistant message

        this.initializeElements();
        this.bindEvents();
        this.initializeSliders();
        this.loadModels();
        this.checkHealth();
    }

    initializeElements() {
        // Main elements
        this.messagesContainer = document.getElementById('messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.modelSelect = document.getElementById('model-select');
        this.refreshModelsButton = document.getElementById('refresh-models');
        this.modelInfoLink = document.getElementById('model-info-link');
        this.clearChatButton = document.getElementById('clear-chat');
        this.statusElement = document.getElementById('status');
        // Modal elements
        this.modelInfoModal = document.getElementById('model-info-modal');
        this.modelInfoTitle = document.getElementById('model-info-title');
        this.modelInfoBody = document.getElementById('model-info-body');
        this.modelInfoClose = document.getElementById('model-info-close');
        this.webSearchCheckbox = document.getElementById('web-search-checkbox');

        // Parameter elements
        this.toggleParamsButton = document.getElementById('toggle-params');
        this.paramsPanel = document.getElementById('params-panel');
        this.paramFilterInfo = document.getElementById('param-filter-info');
        this.resetParamsButton = document.getElementById('reset-params');
        this.savePresetButton = document.getElementById('save-preset');
        this.loadPresetButton = document.getElementById('load-preset');

        // Model search elements
        this.toggleModelSearchBtn = document.getElementById('toggle-model-search-btn');
        this.modelSearchPanel = document.getElementById('model-search-panel');
        this.modelSearchTerm = document.getElementById('model-search-term');
        this.modelSort = document.getElementById('model-sort');
        this.applyModelSearchButton = document.getElementById('apply-model-search');
        this.clearModelSearchButton = document.getElementById('clear-model-search');
        this.searchResultsCount = document.getElementById('search-results-count');

        // Filter elements
        this.inputModalityElements = {
            text: document.getElementById('input-text'),
            image: document.getElementById('input-image'),
            file: document.getElementById('input-file'),
            audio: document.getElementById('input-audio')
        };
        this.outputModalityElements = {
            text: document.getElementById('output-text'),
            image: document.getElementById('output-image')
        };
        // Updated context length slider (single slider for minimum)
        this.contextLengthSlider = document.getElementById('context-length-slider');
        this.contextLengthLabel = document.getElementById('context-length-label');
        this.resetContextButton = document.getElementById('reset-context');

        this.minPriceSlider = document.getElementById('min-price-slider');
        this.maxPriceSlider = document.getElementById('max-price-slider');
        this.minPriceLabel = document.getElementById('min-price-label');
        this.maxPriceLabel = document.getElementById('max-price-label');
        this.resetPriceButton = document.getElementById('reset-price');
        this.freeOnly = document.getElementById('free-only');
        this.searchLimit = document.getElementById('search-limit');
        this.searchOffset = document.getElementById('search-offset');

        // Parameter checkbox elements
        this.parameterCheckboxes = {
            temperature: document.getElementById('param-temperature'),
            top_p: document.getElementById('param-top-p'),
            top_k: document.getElementById('param-top-k'),
            frequency_penalty: document.getElementById('param-frequency-penalty'),
            presence_penalty: document.getElementById('param-presence-penalty'),
            max_tokens: document.getElementById('param-max-tokens'),
            stop: document.getElementById('param-stop'),
            response_format: document.getElementById('param-response-format'),
            tools: document.getElementById('param-tools'),
            tool_choice: document.getElementById('param-tool-choice'),
            reasoning: document.getElementById('param-reasoning'),
            reasoning_exclude: document.getElementById('param-reasoning-exclude')
        };

        // Slider elements with value displays
        this.temperatureSlider = document.getElementById('temperature');
        this.tempValue = document.getElementById('temp-value');
        this.topPSlider = document.getElementById('top-p');
        this.topPValue = document.getElementById('top-p-value');
        this.freqPenaltySlider = document.getElementById('frequency-penalty');
        this.freqPenaltyValue = document.getElementById('freq-penalty-value');
        this.presPenaltySlider = document.getElementById('presence-penalty');
        this.presPenaltyValue = document.getElementById('pres-penalty-value');
    }

    bindEvents() {
        // Main app events
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.messageInput.addEventListener('input', () => this.updateSendButton());
        this.modelSelect.addEventListener('change', (e) => {
            this.currentModel = e.target.value;
            this.updateSendButton();
            this.updateParameterVisibility();
        });
        this.refreshModelsButton.addEventListener('click', () => this.loadModels());
        this.modelInfoLink.addEventListener('click', (e) => {
            e.preventDefault();
            this.openModelInfo();
        });
        this.clearChatButton.addEventListener('click', () => this.clearChat());
        // Modal events
        this.modelInfoClose.addEventListener('click', () => this.closeModelInfo());
        this.modelInfoModal.addEventListener('click', (e) => {
            if (e.target === this.modelInfoModal) {
                this.closeModelInfo();
            }
        });

        // Parameter events
        this.toggleParamsButton.addEventListener('click', () => this.toggleParameters());
        this.resetParamsButton.addEventListener('click', () => this.resetParameters());
        this.savePresetButton.addEventListener('click', () => this.savePreset());
        this.loadPresetButton.addEventListener('click', () => this.loadPreset());

        // Model search events
        this.toggleModelSearchBtn.addEventListener('click', () => this.toggleModelSearch());
        this.applyModelSearchButton.addEventListener('click', () => this.performModelSearch());
        this.clearModelSearchButton.addEventListener('click', () => this.clearModelSearch());

        // Auto-search on enter in search term
        this.modelSearchTerm.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.performModelSearch();
            }
        });

        // Slider events
        this.temperatureSlider.addEventListener('input', () => {
            this.tempValue.textContent = parseFloat(this.temperatureSlider.value).toFixed(1);
        });
        this.topPSlider.addEventListener('input', () => {
            this.topPValue.textContent = parseFloat(this.topPSlider.value).toFixed(2);
        });
        this.freqPenaltySlider.addEventListener('input', () => {
            this.freqPenaltyValue.textContent = parseFloat(this.freqPenaltySlider.value).toFixed(1);
        });
        this.presPenaltySlider.addEventListener('input', () => {
            this.presPenaltyValue.textContent = parseFloat(this.presPenaltySlider.value).toFixed(1);
        });

        // Context length slider events (single slider)
        this.contextLengthSlider.addEventListener('input', () => {
            this.updateContextLabel();
        });

        // Price slider events
        this.minPriceSlider.addEventListener('input', () => {
            this.updatePriceLabels();
            this.ensureMinMaxOrder('price');
        });
        this.maxPriceSlider.addEventListener('input', () => {
            this.updatePriceLabels();
            this.ensureMinMaxOrder('price');
        });

        // Reset button events
        this.resetContextButton.addEventListener('click', () => this.resetContextSlider());
        this.resetPriceButton.addEventListener('click', () => this.resetPriceSliders());
    }

    initializeSliders() {
        // Initialize slider labels with default values
        this.updateContextLabel();
        this.updatePriceLabels();
    }

    updateSendButton() {
        const hasMessage = this.messageInput.value.trim().length > 0;
        const hasModel = this.currentModel && this.currentModel.trim().length > 0;
        this.sendButton.disabled = !hasMessage || !hasModel || this.isStreaming;
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            if (response.ok) {
                this.setStatus('connected', 'Connected');
            } else {
                this.setStatus('error', 'Backend Error');
            }
        } catch (error) {
            this.setStatus('error', 'Connection Failed');
            console.error('Health check failed:', error);
        }
    }

    setStatus(type, message) {
        this.statusElement.className = `status ${type}`;
        this.statusElement.textContent = message;
    }

    async loadModels() {
        try {
            this.modelSelect.innerHTML = '<option value="">Loading models...</option>';
            this.refreshModelsButton.disabled = true;

            const response = await fetch(`${this.baseUrl}/models`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.allModels = data.models || data;
            this.populateModels(this.allModels);

        } catch (error) {
            console.error('Failed to load models:', error);
            this.modelSelect.innerHTML = '<option value="">Failed to load models</option>';
            this.addMessage('error', `Failed to load models: ${error.message}`);
        } finally {
            this.refreshModelsButton.disabled = false;
        }
    }

    populateModels(models) {
        this.modelSelect.innerHTML = '<option value="">Select a model...</option>';

        if (Array.isArray(models)) {
            models.forEach(model => {
                const option = document.createElement('option');
                const modelId = model.id || model.name || model;
                option.value = modelId;
                option.textContent = modelId;
                this.modelSelect.appendChild(option);

                // Store model capabilities
                if (model.supported_parameters) {
                    this.modelCapabilities[modelId] = model.supported_parameters;
                } else {
                    // Fallback: assume all parameters are supported for models without explicit support info
                    this.modelCapabilities[modelId] = 'all';
                }
            });
        }
        this.updateSendButton();
        this.updateParameterVisibility(); // Update parameter visibility after loading models
    }

    async openModelInfo() {
        if (!this.currentModel || !this.currentModel.trim()) {
            this.modelInfoTitle.textContent = 'Model Info';
            this.modelInfoBody.innerHTML = '<p class="muted">Please select a model first.</p>';
            this.modelInfoModal.classList.remove('hidden');
            return;
        }

        this.modelInfoTitle.textContent = this.currentModel;
        this.modelInfoBody.textContent = 'Loading...';
        this.modelInfoModal.classList.remove('hidden');

        try {
            const resp = await fetch(`${this.baseUrl}/models/${encodeURIComponent(this.currentModel)}`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            const info = await resp.json();

            // Store provider parameter union for parameter visibility
            if (info.provider_parameter_union) {
                this.modelCapabilities[this.currentModel] = info.provider_parameter_union;
                // Update parameter visibility when we get fresh model details
                this.updateParameterVisibility();
            }

            const parts = [];
            if (info.description) {
                parts.push(`<p>${this.escapeHtml(info.description)}</p>`);
            }
            const meta = [];
            if (info.context_length) meta.push(`Context: ${info.context_length.toLocaleString()} tokens`);
            if (info.provider_name) meta.push(`Provider: ${this.escapeHtml(info.provider_name)}`);
            if (info.pricing && (info.pricing.prompt || info.pricing.completion)) {
                const prompt = info.pricing.prompt ? `$${info.pricing.prompt}/1M in` : null;
                const comp = info.pricing.completion ? `$${info.pricing.completion}/1M out` : null;
                meta.push(['Pricing', [prompt, comp].filter(Boolean).join(', ')].join(': '));
            }
            if (meta.length) {
                parts.push(`<p class="muted">${meta.map(this.escapeHtml).join(' · ')}</p>`);
            }

            // Show provider parameter union info
            if (info.provider_parameter_union && info.provider_parameter_union.length) {
                parts.push(`<p><strong>Supported Parameters (Union):</strong> ${info.provider_parameter_union.map(this.escapeHtml).join(', ')}</p>`);
            }

            // Show provider parameter intersection if different from union
            if (info.provider_parameter_intersection && info.provider_parameter_intersection.length &&
                JSON.stringify(info.provider_parameter_intersection) !== JSON.stringify(info.provider_parameter_union)) {
                parts.push(`<p><strong>Parameters Supported by ALL Providers:</strong> ${info.provider_parameter_intersection.map(this.escapeHtml).join(', ')}</p>`);
            }

            // Show provider-specific parameter details
            if (info.provider_parameter_details && Object.keys(info.provider_parameter_details).length > 0) {
                const providerItems = Object.entries(info.provider_parameter_details).map(([provider, params]) => {
                    return `<li><strong>${this.escapeHtml(provider)}:</strong> ${params.map(this.escapeHtml).join(', ')}</li>`;
                }).join('');
                parts.push(`<div><strong>Provider-Specific Parameters:</strong><ul style="margin-left:1rem;">${providerItems}</ul></div>`);
            }

            if (info.supported_parameters && Array.isArray(info.supported_parameters) && info.supported_parameters.length) {
                parts.push(`<p><strong>Base Model Parameters:</strong> ${info.supported_parameters.map(this.escapeHtml).join(', ')}</p>`);
            }

            if (info.endpoints && Array.isArray(info.endpoints) && info.endpoints.length) {
                const items = info.endpoints.map(ep => {
                    const name = this.escapeHtml(ep.name || ep.provider || 'Endpoint');
                    const url = this.escapeHtml(ep.url || '');
                    return `<li>${name}${url ? ` — <span class="muted">${url}</span>` : ''}</li>`;
                }).join('');
                parts.push(`<div><strong>Endpoints:</strong><ul style="margin-left:1rem;">${items}</ul></div>`);
            }

            if (parts.length === 0) {
                parts.push('<p class="muted">No details available for this model.</p>');
            }

            this.modelInfoBody.innerHTML = parts.join('');
        } catch (err) {
            console.error('Failed to fetch model info:', err);
            this.modelInfoBody.innerHTML = '<p class="muted">Failed to load model details.</p>';
        }
    }

    closeModelInfo() {
        this.modelInfoModal.classList.add('hidden');
    }

    escapeHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Parameter Methods
    toggleParameters() {
        console.log('Toggle parameters clicked'); // Debug log
        this.paramsPanel.classList.toggle('hidden');
    }

    resetParameters() {
        // Reset sliders to defaults
        this.temperatureSlider.value = '1.0';
        this.tempValue.textContent = '1.0';
        this.topPSlider.value = '1.0';
        this.topPValue.textContent = '1.0';
        this.freqPenaltySlider.value = '0.0';
        this.freqPenaltyValue.textContent = '0.0';
        this.presPenaltySlider.value = '0.0';
        this.presPenaltyValue.textContent = '0.0';

        // Reset all other inputs
        const inputs = this.paramsPanel.querySelectorAll('input:not([type="range"]), select, textarea');
        inputs.forEach(input => {
            if (input.type === 'checkbox') {
                input.checked = false;
            } else if (input.id === 'reasoning-effort') {
                // Reset reasoning effort to default 'high' to match backend default
                input.value = 'high';
            } else {
                input.value = '';
            }
        });
    }

    async updateParameterVisibility() {
        if (!this.currentModel) {
            // No model selected - show all parameters
            this.showAllParameters();
            this.paramFilterInfo.classList.add('hidden');
            return;
        }

        // Try to get provider parameter union from cache first
        let supportedParams = this.modelCapabilities[this.currentModel];

        // If not in cache or if it's the old fallback, fetch fresh model details
        if (!supportedParams || supportedParams === 'all') {
            try {
                console.log(`Fetching model details for parameter visibility: ${this.currentModel}`);
                const resp = await fetch(`${this.baseUrl}/models/${encodeURIComponent(this.currentModel)}`);
                if (resp.ok) {
                    const info = await resp.json();
                    if (info.provider_parameter_union && Array.isArray(info.provider_parameter_union)) {
                        supportedParams = info.provider_parameter_union;
                        // Cache the result
                        this.modelCapabilities[this.currentModel] = supportedParams;
                        console.log(`Cached provider parameter union for ${this.currentModel}:`, supportedParams);
                    }
                }
            } catch (error) {
                console.error('Failed to fetch model details for parameter visibility:', error);
            }
        }

        if (!supportedParams || supportedParams === 'all') {
            // Fallback: model not found or supports all parameters - show everything
            this.showAllParameters();
            this.paramFilterInfo.classList.add('hidden');
            return;
        }

        console.log(`Model ${this.currentModel} provider parameter union:`, supportedParams);

        // Show filter info when filtering parameters
        this.paramFilterInfo.classList.remove('hidden');

        // Define parameter mappings (frontend ID -> OpenRouter parameter name)
        const parameterMappings = {
            // Model behavior parameters
            'temperature': 'temperature',
            'top-p': 'top_p',
            'top-k': 'top_k',
            'frequency-penalty': 'frequency_penalty',
            'presence-penalty': 'presence_penalty',
            'repetition-penalty': 'repetition_penalty',
            'min-p': 'min_p',
            'top-a': 'top_a',
            'seed': 'seed',

            // Output controls
            'max-tokens': 'max_tokens',
            'stop': 'stop',
            'logit-bias': 'logit_bias',
            'logprobs': 'logprobs',
            'top-logprobs': 'top_logprobs',
            'response-format': 'response_format',

            // Function calling
            'tools': 'tools',
            'tool-choice': 'tool_choice',

            // Reasoning controls
            'reasoning-effort': 'reasoning_effort',
            'reasoning-max-tokens': 'reasoning_max_tokens',
            'hide-reasoning': 'reasoning_exclude',  // Maps to backend reasoning_exclude parameter
            'disable-reasoning': 'disable_reasoning'  // Maps to backend disable_reasoning parameter
        };

        // Show/hide parameters based on provider parameter union
        Object.entries(parameterMappings).forEach(([elementId, paramName]) => {
            const paramRow = document.getElementById(elementId)?.closest('.param-row');
            if (paramRow) {
                let shouldShow = false;

                // For reasoning parameters, show if model supports any reasoning capability
                if (['reasoning_effort', 'reasoning_max_tokens', 'reasoning_exclude', 'disable_reasoning'].includes(paramName)) {
                    shouldShow = supportedParams.some(param =>
                        ['reasoning', 'reasoning_effort', 'reasoning_max_tokens', 'reasoning_exclude', 'disable_reasoning'].includes(param)
                    );
                } else {
                    // For other parameters, check if it's in the provider parameter union
                    shouldShow = supportedParams.includes(paramName);
                }

                if (shouldShow) {
                    paramRow.style.display = '';
                    paramRow.classList.remove('param-disabled');
                } else {
                    paramRow.style.display = 'none';
                    paramRow.classList.add('param-disabled');
                }
            }
        });

        // Always show OpenRouter routing controls (these work for all models)
        const alwaysShowIds = ['sort', 'providers', 'max-price', 'fallbacks', 'require-parameters'];
        alwaysShowIds.forEach(id => {
            const paramRow = document.getElementById(id)?.closest('.param-row');
            if (paramRow) {
                paramRow.style.display = '';
                paramRow.classList.remove('param-disabled');
            }
        });

        this.updateParameterGroupVisibility();
    }

    showAllParameters() {
        // Show all parameter rows
        const paramRows = this.paramsPanel.querySelectorAll('.param-row');
        paramRows.forEach(row => {
            row.style.display = '';
            row.classList.remove('param-disabled');
        });

        this.updateParameterGroupVisibility();

        // Hide filter info when showing all parameters
        if (this.paramFilterInfo) {
            this.paramFilterInfo.classList.add('hidden');
        }
    }

    updateParameterGroupVisibility() {
        // Hide parameter groups that have no visible parameters
        const paramGroups = this.paramsPanel.querySelectorAll('.param-group');
        paramGroups.forEach(group => {
            const visibleRows = group.querySelectorAll('.param-row:not([style*="display: none"])');
            if (visibleRows.length === 0) {
                group.style.display = 'none';
            } else {
                group.style.display = '';
            }
        });
    }

    // Context length slider helper methods
    contextSliderToValue(sliderValue) {
        const contextValues = [
            { value: 0, tokens: 0, label: 'Any' },
            { value: 1, tokens: 4000, label: '4K+' },
            { value: 2, tokens: 8000, label: '8K+' },
            { value: 3, tokens: 16000, label: '16K+' },
            { value: 4, tokens: 32000, label: '32K+' },
            { value: 5, tokens: 64000, label: '64K+' },
            { value: 6, tokens: 128000, label: '128K+' },
            { value: 7, tokens: 200000, label: '200K+' },
            { value: 8, tokens: 500000, label: '500K+' },
            { value: 9, tokens: 1000000, label: '1M+' },
            { value: 10, tokens: 2000000, label: '2M+' }
        ];

        const match = contextValues.find(cv => cv.value === parseInt(sliderValue));
        return match || contextValues[contextValues.length - 1];
    }

    // Price slider helper methods
    priceSliderToValue(sliderValue) {
        const priceValues = [
            { value: 0, price: 0, label: 'FREE' },
            { value: 1, price: 0.0001, label: '$0.0001' },
            { value: 2, price: 0.001, label: '$0.001' },
            { value: 3, price: 0.01, label: '$0.01' },
            { value: 4, price: 0.1, label: '$0.1' },
            { value: 5, price: 0.5, label: '$0.5' },
            { value: 6, price: 1.0, label: '$1' },
            { value: 7, price: 2.0, label: '$2' },
            { value: 8, price: 5.0, label: '$5' },
            { value: 9, price: 10.0, label: '$10' },
            { value: 10, price: 999999, label: '$10+' }
        ];

        const match = priceValues.find(pv => pv.value === parseInt(sliderValue));
        return match || priceValues[priceValues.length - 1];
    }

    updateContextLabel() {
        const contextValue = this.contextSliderToValue(this.contextLengthSlider.value);
        this.contextLengthLabel.textContent = contextValue.label;
    }

    updatePriceLabels() {
        const minValue = this.priceSliderToValue(this.minPriceSlider.value);
        const maxValue = this.priceSliderToValue(this.maxPriceSlider.value);

        this.minPriceLabel.textContent = minValue.label;
        this.maxPriceLabel.textContent = maxValue.label;
    }

    ensureMinMaxOrder(type) {
        if (type === 'price') {
            const minVal = parseInt(this.minPriceSlider.value);
            const maxVal = parseInt(this.maxPriceSlider.value);

            if (minVal > maxVal) {
                this.minPriceSlider.value = maxVal;
                this.updatePriceLabels();
            }
        }
    }

    resetContextSlider() {
        this.contextLengthSlider.value = '0';
        this.updateContextLabel();
    }

    resetPriceSliders() {
        this.minPriceSlider.value = '0';
        this.maxPriceSlider.value = '10';
        this.updatePriceLabels();
    }

    getParameters() {
        const params = {};

        const getValue = (id) => {
            const element = document.getElementById(id);
            if (!element) return null;

            if (element.type === 'checkbox') {
                return element.checked ? true : null;
            } else if (element.value && element.value.trim()) {
                if (element.type === 'number' || element.type === 'range') {
                    return parseFloat(element.value);
                }
                return element.value.trim();
            }
            return null;
        };

        // OpenRouter routing & cost controls
        const sort = getValue('sort');
        if (sort) params.sort = sort;

        const providers = getValue('providers');
        if (providers) params.providers = providers;

        const maxPrice = getValue('max-price');
        if (maxPrice !== null) params.max_price = maxPrice;

        const fallbacks = getValue('fallbacks');
        if (fallbacks) params.fallbacks = fallbacks;

        const requireParameters = getValue('require-parameters');
        if (requireParameters) params.require_parameters = requireParameters;

        // Model behavior parameters
        const temperature = getValue('temperature');
        if (temperature !== null && temperature !== 1.0) params.temperature = temperature;

        const topP = getValue('top-p');
        if (topP !== null && topP !== 1.0) params.top_p = topP;

        const topK = getValue('top-k');
        if (topK !== null) params.top_k = topK;

        const frequencyPenalty = getValue('frequency-penalty');
        if (frequencyPenalty !== null && frequencyPenalty !== 0.0) params.frequency_penalty = frequencyPenalty;

        const presencePenalty = getValue('presence-penalty');
        if (presencePenalty !== null && presencePenalty !== 0.0) params.presence_penalty = presencePenalty;

        const repetitionPenalty = getValue('repetition-penalty');
        if (repetitionPenalty !== null) params.repetition_penalty = repetitionPenalty;

        const minP = getValue('min-p');
        if (minP !== null) params.min_p = minP;

        const topA = getValue('top-a');
        if (topA !== null) params.top_a = topA;

        const seed = getValue('seed');
        if (seed !== null) params.seed = seed;

        // Output controls
        const maxTokens = getValue('max-tokens');
        if (maxTokens !== null) params.max_tokens = maxTokens;

        const stop = getValue('stop');
        if (stop) params.stop = stop;

        const logitBias = getValue('logit-bias');
        if (logitBias) params.logit_bias = logitBias;

        const logprobs = getValue('logprobs');
        if (logprobs) params.logprobs = logprobs;

        const topLogprobs = getValue('top-logprobs');
        if (topLogprobs !== null) params.top_logprobs = topLogprobs;

        const responseFormat = getValue('response-format');
        if (responseFormat) params.response_format = responseFormat;

        // Function calling
        const tools = getValue('tools');
        if (tools) params.tools = tools;

        const toolChoice = getValue('tool-choice');
        if (toolChoice) params.tool_choice = toolChoice;

        // Reasoning controls
        // Checkbox id=disable-reasoning maps to backend disable_reasoning param (full disable of reasoning path)
        const disableReasoningEl = document.getElementById('disable-reasoning');
        const disableReasoning = disableReasoningEl?.checked;
        if (disableReasoning) {
            params.disable_reasoning = true;
        } else {
            const reasoningEffort = getValue('reasoning-effort');
            if (reasoningEffort) params.reasoning_effort = reasoningEffort;

            const reasoningMaxTokens = getValue('reasoning-max-tokens');
            if (reasoningMaxTokens !== null) params.reasoning_max_tokens = reasoningMaxTokens;

            // Hide reasoning checkbox maps to reasoning_exclude parameter
            const hideReasoning = getValue('hide-reasoning');
            if (hideReasoning) params.reasoning_exclude = hideReasoning;
        }

        return params;
    }

    buildUrlWithParams() {
        const params = this.getParameters();
        const urlParams = new URLSearchParams();

        Object.entries(params).forEach(([key, value]) => {
            urlParams.append(key, value.toString());
        });

        const paramString = urlParams.toString();
        return `${this.baseUrl}/chat${paramString ? '?' + paramString : ''}`;
    }

    savePreset() {
        const preset = this.getParameters();
        const presetName = prompt('Enter a name for this preset:');
        if (presetName) {
            localStorage.setItem(`chat-preset-${presetName}`, JSON.stringify(preset));
            alert(`Preset "${presetName}" saved!`);
        }
    }

    loadPreset() {
        const presetName = prompt('Enter the name of the preset to load:');
        if (presetName) {
            const preset = localStorage.getItem(`chat-preset-${presetName}`);
            if (preset) {
                this.applyParameters(JSON.parse(preset));
                alert(`Preset "${presetName}" loaded!`);
            } else {
                alert(`Preset "${presetName}" not found.`);
            }
        }
    }

    applyParameters(params) {
        Object.entries(params).forEach(([key, value]) => {
            if (value === null || value === undefined) return;

            let elementId = key.replace(/_/g, '-');
            let element = document.getElementById(elementId);
            // Map backend parameters to frontend element IDs
            if (!element && key === 'disable_reasoning') {
                element = document.getElementById('disable-reasoning');
            }
            if (!element && key === 'reasoning_exclude') {
                element = document.getElementById('hide-reasoning');
            }
            if (!element) return;

            if (element.type === 'checkbox') {
                element.checked = Boolean(value);
            } else {
                element.value = value;
                // Update slider displays
                if (element.type === 'range') {
                    const valueDisplay = document.getElementById(`${elementId.replace('-', '')}-value`) ||
                        document.getElementById(`${elementId}-value`);
                    if (valueDisplay) {
                        valueDisplay.textContent = parseFloat(value).toFixed(elementId.includes('penalty') ? 1 :
                            elementId.includes('top-p') ? 2 : 1);
                    }
                }
            }
        });
    }

    // Model Search Methods
    toggleModelSearch() {
        console.log('Toggle model search clicked');
        this.modelSearchPanel.classList.toggle('hidden');
    }

    getModelSearchFilters() {
        const filters = {};

        // Search term
        if (this.modelSearchTerm.value.trim()) {
            filters.search_term = this.modelSearchTerm.value.trim();
        }

        // Input modalities
        const inputModalities = [];
        Object.entries(this.inputModalityElements).forEach(([key, element]) => {
            if (element && element.checked) {
                inputModalities.push(key);
            }
        });
        if (inputModalities.length > 0) {
            filters.input_modalities = inputModalities.join(',');
        }

        // Output modalities
        const outputModalities = [];
        Object.entries(this.outputModalityElements).forEach(([key, element]) => {
            if (element && element.checked) {
                outputModalities.push(key);
            }
        });
        if (outputModalities.length > 0) {
            filters.output_modalities = outputModalities.join(',');
        }

        // Context length from single slider (minimum context length)
        const contextValue = this.contextSliderToValue(this.contextLengthSlider.value);

        if (parseInt(this.contextLengthSlider.value) > 0) {
            filters.min_context_length = contextValue.tokens;
        }

        // Price filters from sliders
        if (this.freeOnly.checked) {
            filters.free_only = true;
        } else {
            const minPriceValue = this.priceSliderToValue(this.minPriceSlider.value);
            const maxPriceValue = this.priceSliderToValue(this.maxPriceSlider.value);

            if (parseInt(this.minPriceSlider.value) > 0) {
                filters.min_price = minPriceValue.price;
            }
            if (parseInt(this.maxPriceSlider.value) < 10) {
                filters.max_price = maxPriceValue.price;
            }
        }

        // Supported parameters - collect checked parameter checkboxes
        const selectedParams = [];
        Object.entries(this.parameterCheckboxes).forEach(([paramName, element]) => {
            if (element && element.checked) {
                selectedParams.push(paramName);
            }
        });
        if (selectedParams.length > 0) {
            filters.supported_parameters = selectedParams.join(',');
        }

        // Sorting
        filters.sort = this.modelSort.value || 'newest';

        // Pagination
        filters.limit = parseInt(this.searchLimit.value) || 50;
        filters.offset = parseInt(this.searchOffset.value) || 0;

        return filters;
    }

    async performModelSearch() {
        try {
            this.applyModelSearchButton.disabled = true;
            this.applyModelSearchButton.textContent = '🔄 Searching...';

            const filters = this.getModelSearchFilters();
            console.log('Searching models with filters:', filters);

            // Build URL with search parameters
            const url = new URL(`${this.baseUrl}/models/search`);
            Object.entries(filters).forEach(([key, value]) => {
                if (value !== null && value !== undefined && value !== '') {
                    url.searchParams.append(key, value.toString());
                }
            });

            const response = await fetch(url.toString());
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Search results:', data);

            // Update search results info
            this.updateSearchResultsInfo(data);

            // Populate models with search results
            this.populateModels(data.models);

        } catch (error) {
            console.error('Model search failed:', error);
            this.addMessage('error', `Model search failed: ${error.message}`);
            this.searchResultsCount.textContent = 'Search failed';
        } finally {
            this.applyModelSearchButton.disabled = false;
            this.applyModelSearchButton.textContent = '🔍 Search Models';
        }
    }

    updateSearchResultsInfo(data) {
        const { models, total_count, filters_applied, sort_applied } = data;

        let infoText = `Found ${total_count} models`;
        if (filters_applied.offset > 0) {
            infoText += ` (showing from ${filters_applied.offset + 1})`;
        }
        if (sort_applied && sort_applied !== 'newest') {
            infoText += ` sorted by ${sort_applied.replace(/_/g, ' ')}`;
        }

        this.searchResultsCount.textContent = infoText;
        this.searchResultsCount.style.color = total_count > 0 ? '#10b981' : '#ef4444';
    }

    clearModelSearch() {
        // Clear all search form fields
        this.modelSearchTerm.value = '';
        this.modelSort.value = 'newest';

        // Clear input modality checkboxes
        Object.values(this.inputModalityElements).forEach(element => {
            if (element) element.checked = false;
        });

        // Clear output modality checkboxes
        Object.values(this.outputModalityElements).forEach(element => {
            if (element) element.checked = false;
        });

        // Clear context length slider
        this.resetContextSlider();

        // Clear price filters
        this.freeOnly.checked = false;
        this.resetPriceSliders();

        // Clear parameter checkboxes
        Object.values(this.parameterCheckboxes).forEach(element => {
            if (element) element.checked = false;
        });

        // Reset pagination
        this.searchLimit.value = '50';
        this.searchOffset.value = '0';

        // Reset search results info
        this.searchResultsCount.textContent = 'No search performed yet';
        this.searchResultsCount.style.color = '#94a3b8';

        // Reload all models
        this.loadModels();
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        if (!this.currentModel) {
            this.addMessage('error', 'Please select a model before sending a message.');
            return;
        }

        // Add user message
        this.addMessage('user', message);

        // Add web search indicator if enabled
        if (this.webSearchCheckbox.checked) {
            this.addMessage('system', '🌐 Web search enabled for this query');
        }

        this.messageInput.value = '';
        this.updateSendButton();

        // Convert messages to backend format
        const history = this.messages.map(msg => ({
            role: msg.role === 'assistant' ? 'model' : msg.role,
            content: msg.content
        })).filter(msg => msg.role === 'user' || msg.role === 'model');

        // Prepare request body
        const requestBody = {
            history: history,
            message: message,
            model: this.currentModel || undefined,
            web_search: this.webSearchCheckbox.checked
        };

        this.isStreaming = true;
        this.updateSendButton();

        try {
            // Create a unique id for this assistant turn
            const messageId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
            await this.handleStreamingResponse(requestBody, messageId);
        } catch (error) {
            console.error('Chat request failed:', error);
            this.addMessage('error', `Request failed: ${error.message}`);
        } finally {
            this.isStreaming = false;
            this.updateSendButton();
        }
    }

    async handleStreamingResponse(requestBody, messageId) {
        const url = this.buildUrlWithParams();
        console.log('Sending request to:', url); // Debug log

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        // Add typing indicator
        const typingIndicator = this.addTypingIndicator();
        let messageElement = null;
        let assistantMessage = '';

        let reasoningContent = '';

        try {
            await this.processServerSentEvents(response,
                // Content handler
                (textChunk) => {
                    // Handle each text chunk from content events
                    assistantMessage += textChunk;

                    // Remove typing indicator on first chunk
                    if (typingIndicator && typingIndicator.parentNode) {
                        typingIndicator.remove();
                    }

                    // Create or update message element
                    if (!messageElement) {
                        messageElement = this.addMessage('assistant', textChunk, { messageId });
                    } else {
                        this.appendToMessage(messageElement, textChunk);
                    }
                },
                // Reasoning handler
                (reasoningChunk) => {
                    // Handle reasoning tokens - model's thinking process
                    console.log('🧠 Reasoning chunk received in handler:', reasoningChunk);

                    // Safe handling for structured reasoning data
                    const text = typeof reasoningChunk === 'string'
                        ? reasoningChunk
                        : (reasoningChunk?.text || reasoningChunk?.delta || JSON.stringify(reasoningChunk));
                    reasoningContent += text;

                    // Ensure message element exists and update reasoning
                    if (!messageElement) {
                        messageElement = this.addMessage('assistant', '', { messageId });
                        console.log('🧠 Created new message element for reasoning');
                    }

                    console.log('🧠 Updating reasoning content, total length:', reasoningContent.length);
                    this.updateReasoningContent(messageElement, reasoningContent);
                },
                // Usage handler (per-stream, at end)
                (usageData) => {
                    // Store per-message usage
                    this.usageByMessageId.set(messageId, usageData);
                    this.currentUsageData = usageData; // keep latest for compatibility
                    // Update the usage link state for this message
                    const msgEl = this.messagesContainer.querySelector(`.message.assistant[data-message-id="${messageId}"]`);
                    if (msgEl) {
                        const link = msgEl.querySelector('.usage-details-link');
                        if (link) {
                            link.style.opacity = '1';
                            link.textContent = '📊 Usage Details';
                        }
                    }
                },
                // Done handler (optional)
                () => {
                    // No-op for now
                }
            );

            // Store the complete message
            if (assistantMessage) {
                this.messages.push({ role: 'assistant', content: assistantMessage });
            }

        } catch (error) {
            if (typingIndicator && typingIndicator.parentNode) {
                typingIndicator.remove();
            }
            throw error;
        }
    }

    async processServerSentEvents(response, onTextChunk, onReasoningChunk, onUsage, onDone) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                // Decode the chunk and add to buffer
                buffer += decoder.decode(value, { stream: true });

                // Process complete SSE events from buffer
                let eventEnd;
                while ((eventEnd = buffer.indexOf('\n\n')) !== -1) {
                    const eventData = buffer.slice(0, eventEnd);
                    buffer = buffer.slice(eventEnd + 2);

                    if (eventData.trim()) {
                        console.log('Processing SSE event:', eventData); // Debug log
                        this.handleSSEEvent(eventData, onTextChunk, onReasoningChunk, onUsage, onDone);
                    }
                }
            }

            // Process any remaining data in buffer
            if (buffer.trim()) {
                console.log('Processing remaining buffer:', buffer); // Debug log
                this.handleSSEEvent(buffer, onTextChunk, onReasoningChunk, onUsage, onDone);
            }

        } finally {
            reader.releaseLock();
        }
    }

    handleSSEEvent(eventData, onTextChunk, onReasoningChunk, onUsage, onDone) {
        const lines = eventData.split('\n').map(line => line.trim()).filter(line => line);
        let eventType = '';
        let data = '';

        // Parse SSE event format
        for (const line of lines) {
            if (line.startsWith('event:')) {
                eventType = line.substring(6).trim();
            } else if (line.startsWith('data:')) {
                data = line.substring(5).trim();
            }
        }

        // Handle different event types
        try {
            console.log('SSE Event Type:', eventType, 'Data:', data); // Debug log
            switch (eventType) {
                case 'content':
                    // Parse JSON data and extract text
                    if (data) {
                        const contentData = JSON.parse(data);
                        if (contentData.text) {
                            console.log('Calling onTextChunk with:', contentData.text); // Debug log
                            onTextChunk(contentData.text);
                        }
                    }
                    break;

                case 'reasoning':
                    // Handle reasoning tokens - model's thinking process
                    console.log('🧠 Received reasoning event:', data);
                    if (data && onReasoningChunk) {
                        const reasoningData = JSON.parse(data);
                        if (reasoningData.text) {
                            console.log('🧠 Processing reasoning text:', reasoningData.text);
                            onReasoningChunk(reasoningData.text);
                        }
                    }
                    break;

                case 'usage':
                    // Handle usage information
                    console.log('📊 Received usage event:', data);
                    if (data) {
                        const usageData = JSON.parse(data);
                        this.currentUsageData = usageData; // keep latest for compatibility
                        if (typeof onUsage === 'function') {
                            onUsage(usageData);
                        }
                        console.log('📊 Stored usage data (latest):', usageData);
                    }
                    break;

                case 'done':
                    // Stream completion - add usage details link if we have usage data
                    console.log('Stream completed');
                    if (typeof onDone === 'function') onDone();
                    if (this.currentUsageData) {
                        console.log('📊 Stream completed with usage data:', this.currentUsageData);
                    }
                    break;

                case 'error':
                    // Handle error events
                    if (data) {
                        const errorData = JSON.parse(data);
                        throw new Error(errorData.error || 'Stream error occurred');
                    }
                    break;

                default:
                    // Log unknown event types for debugging
                    console.log(`Unknown SSE event type: ${eventType}`, data);
            }
        } catch (parseError) {
            console.error('Error parsing SSE event:', parseError, 'Event data:', eventData);
            // Continue processing other events even if one fails to parse
        }
    }

    addMessage(role, content, options = {}) {
        const { messageId = null } = options;
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        if (role === 'assistant' && messageId) {
            messageDiv.setAttribute('data-message-id', messageId);
        }

        // Add reasoning section for assistant messages (will be populated later if needed)
        if (role === 'assistant') {
            const reasoningDiv = document.createElement('div');
            reasoningDiv.className = 'reasoning-section hidden';

            const reasoningHeader = document.createElement('div');
            reasoningHeader.className = 'reasoning-header';
            reasoningHeader.innerHTML = '<span class="reasoning-icon">🧠</span> <span class="reasoning-label">Model Thinking</span> <span class="reasoning-toggle">▼</span>';

            const reasoningContent = document.createElement('div');
            reasoningContent.className = 'reasoning-content';

            // Make header clickable to toggle reasoning visibility
            reasoningHeader.addEventListener('click', () => {
                const isVisible = !reasoningContent.classList.contains('hidden');
                reasoningContent.classList.toggle('hidden');
                reasoningHeader.querySelector('.reasoning-toggle').textContent = isVisible ? '▶' : '▼';
            });

            reasoningDiv.appendChild(reasoningHeader);
            reasoningDiv.appendChild(reasoningContent);
            messageDiv.appendChild(reasoningDiv);
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        metaDiv.textContent = new Date().toLocaleTimeString();

        // Add usage details link for assistant messages
        if (role === 'assistant') {
            const usageLink = document.createElement('a');
            usageLink.href = '#';
            usageLink.className = 'usage-details-link';
            usageLink.textContent = '📊 Usage Details';
            usageLink.style.cssText = 'margin-left: 10px; color: #007bff; text-decoration: none; font-size: 0.9em; opacity: 0.6; transition: opacity 0.3s ease;';
            usageLink.addEventListener('click', (e) => {
                e.preventDefault();
                if (messageId) {
                    this.showUsageModalFor(messageId);
                } else {
                    // Fallback to latest if no id
                    this.showUsageModal();
                }
            });
            metaDiv.appendChild(usageLink);

            // Update link visibility when usage data becomes available
            const updateUsageLink = () => {
                if (messageId && this.usageByMessageId.has(messageId)) {
                    usageLink.style.opacity = '1';
                    usageLink.textContent = '📊 Usage Details';
                } else {
                    usageLink.style.opacity = '0.6';
                    usageLink.textContent = '📊 Usage Details (pending)';
                }
            };

            // Check immediately and set up interval to check for usage data
            updateUsageLink();
            const checkInterval = setInterval(() => {
                updateUsageLink();
                if (messageId && this.usageByMessageId.has(messageId)) {
                    clearInterval(checkInterval);
                }
            }, 100);
        }

        messageDiv.appendChild(contentDiv);
        if (role !== 'system' && role !== 'error') {
            messageDiv.appendChild(metaDiv);
        }

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        // Store user messages
        if (role === 'user') {
            this.messages.push({ role: 'user', content });
        }

        return messageDiv;
    }

    appendToMessage(messageElement, newContent) {
        const contentDiv = messageElement.querySelector('.message-content');
        if (contentDiv) {
            contentDiv.textContent += newContent;
            this.scrollToBottom();
        }
    }

    updateReasoningContent(messageElement, reasoningContent) {
        console.log('🧠 updateReasoningContent called with content length:', reasoningContent?.length);
        const reasoningSection = messageElement.querySelector('.reasoning-section');
        const reasoningContentDiv = messageElement.querySelector('.reasoning-content');

        console.log('🧠 Found reasoning elements:', {
            section: !!reasoningSection,
            contentDiv: !!reasoningContentDiv,
            hasContent: !!reasoningContent
        });

        if (reasoningSection && reasoningContentDiv && reasoningContent) {
            // Show the reasoning section if we have content
            reasoningSection.classList.remove('hidden');
            reasoningContentDiv.textContent = reasoningContent;
            console.log('🧠 Reasoning section updated and shown');

            // Auto-scroll to keep content visible
            this.scrollToBottom();
        }
    }

    addTypingIndicator() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant typing-indicator';

        const contentDiv = document.createElement('div');
        contentDiv.innerHTML = `
            Typing
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;

        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        return messageDiv;
    }

    clearChat() {
        // Keep only the system welcome message
        const systemMessage = this.messagesContainer.querySelector('.message.system');
        this.messagesContainer.innerHTML = '';
        if (systemMessage) {
            this.messagesContainer.appendChild(systemMessage);
        }
        this.messages = [];
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    showUsageModal() {
        if (!this.currentUsageData) {
            alert('No usage data available for this message.');
            return;
        }

        const usage = this.currentUsageData;

        // Format cost values for display
        const formatCost = (cost) => {
            if (cost === null || cost === undefined) return 'N/A';
            return `$${cost.toFixed(8)}`;
        };

        const formatTokens = (tokens) => {
            if (tokens === null || tokens === undefined) return 'N/A';
            return tokens.toLocaleString();
        };

        const formatRate = (rate) => {
            if (rate === null || rate === undefined) return 'N/A';
            try {
                return `${Number(rate).toFixed(2)} tok/s`;
            } catch {
                return 'N/A';
            }
        };

        // Create modal HTML with clear data source separation
        const modalHTML = `
            <div id="usage-modal" class="modal-overlay">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>📊 Usage Details</h3>
                        <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="usage-section">
                            <h4>🌐 OpenRouter Authoritative Data</h4>
                            <p class="data-source-note">This data comes directly from OpenRouter's response and represents the authoritative billing and model information.</p>

                            <div class="subsection">
                                <h5>💰 Cost Information</h5>
                                <div class="usage-grid">
                                    <div class="usage-item">
                                        <span class="usage-label">Total Cost:</span>
                                        <span class="usage-value cost">${formatCost(usage.cost)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Prompt Cost:</span>
                                        <span class="usage-value">${formatCost(usage.prompt_cost)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Completion Cost:</span>
                                        <span class="usage-value">${formatCost(usage.completion_cost)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">BYOK:</span>
                                        <span class="usage-value">${usage.is_byok ? 'Yes' : 'No'}</span>
                                    </div>
                                </div>
                            </div>

                            <div class="subsection">
                                <h5>📊 Token Usage</h5>
                                <div class="usage-grid">
                                    <div class="usage-item">
                                        <span class="usage-label">Prompt Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.prompt_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Completion Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.completion_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Total Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.total_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Tokens/sec:</span>
                                        <span class="usage-value">${formatRate(usage.tokens_per_second)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Cached Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.cached_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Reasoning Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.reasoning_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Audio Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.audio_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Image Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.image_tokens)}</span>
                                    </div>
                                </div>
                            </div>

                            <div class="subsection">
                                <h5>🤖 Model & Provider Information</h5>
                                <div class="usage-grid">
                                    <div class="usage-item">
                                        <span class="usage-label">Requested Model:</span>
                                        <span class="usage-value">${usage.model || 'N/A'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Actual Model Used:</span>
                                        <span class="usage-value ${usage.actual_model ? 'actual-model' : 'na-value'}">${usage.actual_model || 'Not provided by OpenRouter'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Provider:</span>
                                        <span class="usage-value">${usage.provider || 'N/A'}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="usage-section">
                            <h4>📱 Application Metrics</h4>
                            <p class="data-source-note">This data is collected by our application for performance monitoring and observability.</p>

                            <div class="subsection">
                                <h5>⏱️ Performance Metrics</h5>
                                <div class="usage-grid">
                                    <div class="usage-item">
                                        <span class="usage-label">Request Duration:</span>
                                        <span class="usage-value">${usage.duration_ms || 'N/A'}ms</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Content Events:</span>
                                        <span class="usage-value">${usage.content_events || 'N/A'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Reasoning Events:</span>
                                        <span class="usage-value">${usage.reasoning_events || 'N/A'}</span>
                                    </div>
                                </div>
                            </div>

                            <div class="subsection">
                                <h5>🔧 Request Configuration</h5>
                                <div class="usage-grid">
                                    <div class="usage-item">
                                        <span class="usage-label">Provider Preference:</span>
                                        <span class="usage-value">${usage.routing?.providers?.join(', ') || 'None'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Fallback Models:</span>
                                        <span class="usage-value">${usage.routing?.fallbacks?.join(', ') || 'None'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Sort Strategy:</span>
                                        <span class="usage-value">${usage.routing?.sort || 'Default'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Max Price Limit:</span>
                                        <span class="usage-value">${usage.routing?.max_price ? formatCost(usage.routing.max_price) : 'None'}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add modal to page
        document.body.insertAdjacentHTML('beforeend', modalHTML);

        // Add click outside to close
        const modal = document.getElementById('usage-modal');
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    // New: show usage modal for a specific assistant message
    showUsageModalFor(messageId) {
        const usage = this.usageByMessageId.get(messageId);
        if (!usage) {
            alert('No usage data available yet for this message.');
            return;
        }

        // Format cost values for display
        const formatCost = (cost) => {
            if (cost === null || cost === undefined) return 'N/A';
            return `$${cost.toFixed(8)}`;
        };

        const formatTokens = (tokens) => {
            if (tokens === null || tokens === undefined) return 'N/A';
            return tokens.toLocaleString();
        };

        const formatRate = (rate) => {
            if (rate === null || rate === undefined) return 'N/A';
            try {
                return `${Number(rate).toFixed(2)} tok/s`;
            } catch {
                return 'N/A';
            }
        };

        const modalId = `usage-modal-${messageId}`;
        const modalHTML = `
            <div id="${modalId}" class="modal-overlay">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>📊 Usage Details</h3>
                        <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="usage-section">
                            <h4>🌐 OpenRouter Authoritative Data</h4>
                            <p class="data-source-note">This data comes directly from OpenRouter's response and represents the authoritative billing and model information.</p>

                            <div class="subsection">
                                <h5>💰 Cost Information</h5>
                                <div class="usage-grid">
                                    <div class="usage-item">
                                        <span class="usage-label">Total Cost:</span>
                                        <span class="usage-value cost">${formatCost(usage.cost)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Prompt Cost:</span>
                                        <span class="usage-value">${formatCost(usage.prompt_cost)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Completion Cost:</span>
                                        <span class="usage-value">${formatCost(usage.completion_cost)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">BYOK:</span>
                                        <span class="usage-value">${usage.is_byok ? 'Yes' : 'No'}</span>
                                    </div>
                                </div>
                            </div>

                            <div class="subsection">
                                <h5>📊 Token Usage</h5>
                                <div class="usage-grid">
                                    <div class="usage-item">
                                        <span class="usage-label">Prompt Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.prompt_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Completion Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.completion_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Total Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.total_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Tokens/sec:</span>
                                        <span class="usage-value">${formatRate(usage.tokens_per_second)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Cached Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.cached_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Reasoning Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.reasoning_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Audio Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.audio_tokens)}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Image Tokens:</span>
                                        <span class="usage-value">${formatTokens(usage.image_tokens)}</span>
                                    </div>
                                </div>
                            </div>

                            <div class="subsection">
                                <h5>🤖 Model & Provider Information</h5>
                                <div class="usage-grid">
                                    <div class="usage-item">
                                        <span class="usage-label">Requested Model:</span>
                                        <span class="usage-value">${usage.model || 'N/A'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Actual Model Used:</span>
                                        <span class="usage-value ${usage.actual_model ? 'actual-model' : 'na-value'}">${usage.actual_model || 'Not provided by OpenRouter'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Provider:</span>
                                        <span class="usage-value">${usage.provider || 'N/A'}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="usage-section">
                            <h4>📱 Application Metrics</h4>
                            <p class="data-source-note">This data is collected by our application for performance monitoring and observability.</p>

                            <div class="subsection">
                                <h5>⏱️ Performance Metrics</h5>
                                <div class="usage-grid">
                                    <div class="usage-item">
                                        <span class="usage-label">Request Duration:</span>
                                        <span class="usage-value">${usage.duration_ms || 'N/A'}ms</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Content Events:</span>
                                        <span class="usage-value">${usage.content_events || 'N/A'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Reasoning Events:</span>
                                        <span class="usage-value">${usage.reasoning_events || 'N/A'}</span>
                                    </div>
                                </div>
                            </div>

                            <div class="subsection">
                                <h5>🔧 Request Configuration</h5>
                                <div class="usage-grid">
                                    <div class="usage-item">
                                        <span class="usage-label">Provider Preference:</span>
                                        <span class="usage-value">${usage.routing?.providers?.join(', ') || 'None'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Fallback Models:</span>
                                        <span class="usage-value">${usage.routing?.fallbacks?.join(', ') || 'None'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Sort Strategy:</span>
                                        <span class="usage-value">${usage.routing?.sort || 'Default'}</span>
                                    </div>
                                    <div class="usage-item">
                                        <span class="usage-label">Max Price Limit:</span>
                                        <span class="usage-value">${usage.routing?.max_price ? formatCost(usage.routing.max_price) : 'None'}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHTML);

        const modal = document.getElementById(modalId);
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing ChatApp'); // Debug log
    new ChatApp();
});
