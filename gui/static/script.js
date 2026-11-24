// Global state variables
let isDarkMode = false; // Tracks whether the system prefers dark mode (used for theming)
let currentSlide = 0;   // Index of the currently displayed "slide" (e.g., subject/night)
let totalSlides = 0;    // Total number of available slides (read from total_slides.json)
let currentIndex = 0;   // Initial slide index used when loading data for the first time
let expandedModelsGlobal = []; // Expanded model+lead tags (e.g. ['usleep_eeg', 'deepresnet_eog', 'ensemble'])

// Preview one example EDF's expected output using the current form selections.
function renderOutputPreview() {
    const tree = document.getElementById('output-preview-tree');
    const note = document.getElementById('output-preview-note');
    const folder = document.getElementById('folderName').value.trim() || 'patient_batch_01';
    const models = ['deepresnet', 'sleeptransformer', 'usleep'].filter(id => document.getElementById(id).checked);
    const leads = getCheckedChannels().map(lead => lead === 'emg' ? 'chin' : lead);
    const software = document.getElementById('review-software').value;
    const resolution = models.length === 1 && models[0] === 'usleep'
        ? RESOLUTION_VALUES[Number(document.getElementById('resolution-slider').value)] : 30;
    const node = (label, children = []) => ({ label, children });
    const importFolder = () => software && software !== 'none' ? [node(`${software}/`,
        (software === 'somnomedics'
            ? ['recording_PRED_somno.txt', 'recording_PRED_somno_conf.txt']
            : ['recording_PRED_compumedics.txt']).map(name => node(name)))] : [];
    const predictionFiles = (majority = false) => [
        node('recording_PRED.npy'), node('recording_PRED.annot'),
        ...(majority ? [node('recording_VALID.npy')] : []),
        ...importFolder(),
    ];
    const modelFolders = models.map(model => node(`${model}/`, [node(`${resolution}s/`, [
        ...leads.map(lead => node(`_${lead}/`, [
            node('majority/', predictionFiles(true)),
            node('<recognized-channel>/', predictionFiles()),
        ])),
        ...(leads.length > 1 ? [node('_all/', [node('majority/', predictionFiles(true))])] : []),
    ])]));
    const ensembleFolders = document.getElementById('ensemble').checked && leads.length
        ? [node('ensemble/', [node('30s/', [
            ...leads.map(lead => node(`_${lead}/`, [
                node('soft-voting/', predictionFiles()),
                node('<recognized-channel>/', predictionFiles()),
            ])),
            ...(leads.length > 1 ? [node('_all/', [node('soft-voting/', [
                ...predictionFiles(),
            ])])] : []),
        ])])] : [];
    const root = node(`${folder}/`, [
        node('status.json'), node('sleep_model_extraction.log'), node('recording.h5'),
        ...modelFolders, ...ensembleFolders,
    ]);
    const lines = [root.label];
    const walk = (items, prefix) => items.forEach((item, index) => {
        const last = index === items.length - 1;
        lines.push(`${prefix}${last ? '└── ' : '├── '}${item.label}`);
        walk(item.children, prefix + (last ? '    ' : '│   '));
    });
    walk(root.children, '');
    tree.textContent = lines.join('\n');
    note.textContent =
        'Each <recognized-channel> folder represents one or more channels found in the EDF. Files named recording use the uploaded EDF stem. Annotation and software import files appear when export succeeds.';
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-bs-toggle="popover"]').forEach(element => new bootstrap.Popover(element));
    document.querySelectorAll('.step-toggle').forEach(button => {
        const details = document.querySelector(button.getAttribute('data-bs-target'));
        details.addEventListener('shown.bs.collapse', () => { button.textContent = 'Close'; });
        details.addEventListener('hidden.bs.collapse', () => { button.textContent = 'Open'; });
    });
});

// Reveal each step only after the preceding required input is complete.
document.addEventListener('DOMContentLoaded', () => {
    let preprocessingConfirmed = false;
    const updateSteps = () => {
        const hasFiles = document.getElementById('edf-files').files.length > 0;
        const hasModalities = getCheckedChannels().length > 0;
        const hasModels = ['deepresnet', 'sleeptransformer', 'usleep'].some(id => document.getElementById(id).checked);
        const hasFolder = document.getElementById('folderName').value.trim().length > 0;
        const hasFilterableSignal = document.getElementById('modality-eeg').checked || document.getElementById('modality-eog').checked;
        ['apply-notch', 'apply-bandpass'].forEach(id => {
            const control = document.getElementById(id);
            control.disabled = !hasFilterableSignal;
            if (!hasFilterableSignal) control.checked = false;
        });
        // There are no children EMG weights, so EMG and Children exclude each other.
        const emg = document.getElementById('modality-emg');
        const children = document.getElementById('model-set-child');
        if (emg.checked && children.checked) document.getElementById('model-set-adult').checked = true;
        children.disabled = emg.checked;
        emg.disabled = children.checked;
        document.getElementById('children-emg-hint').classList.toggle('d-none', !emg.checked);
        document.getElementById('emg-children-hint').classList.toggle('d-none', !children.checked);
        document.getElementById('channels-card').classList.toggle('d-none', !hasFiles);
        const ready = preprocessingConfirmed;
        document.getElementById('preprocessing-card').classList.toggle('d-none', !hasFiles || !hasModalities);
        document.getElementById('algorithms-select').classList.toggle('d-none', !hasFiles || !hasModalities || !ready);
        document.getElementById('output-card').classList.toggle('d-none', !hasFiles || !hasModalities || !ready || !hasModels);
        document.getElementById('form-actions').classList.toggle('d-none', !hasFiles || !hasModalities || !ready || !hasModels || !hasFolder);
        renderOutputPreview();
    };
    ['change', 'input'].forEach(event => document.getElementById('uploadForm').addEventListener(event, updateSteps));
    document.getElementById('preprocessing-card').addEventListener('input', () => {
        preprocessingConfirmed = false;
        updateSteps();
    });
    document.getElementById('preprocessing-card').addEventListener('change', () => {
        preprocessingConfirmed = false;
        updateSteps();
    });
    document.getElementById('preprocessing-continue').addEventListener('click', () => {
        const low = Number(document.getElementById('bandpass-low').value);
        const high = Number(document.getElementById('bandpass-high').value);
        const percentInput = document.getElementById('max-bad-epoch-percent');
        const validBandpass = !document.getElementById('apply-bandpass').checked ||
            (low > 0 && high > low);
        const validPercent = !document.getElementById('apply-channel-qc').checked ||
            (percentInput.value !== '' && Number(percentInput.value) >= 0 && Number(percentInput.value) <= 100);
        const error = document.getElementById('preprocessing-error');
        error.textContent = !validBandpass ? 'Enter valid bandpass cutoffs.' : 'Enter a percentage between 0 and 100.';
        error.classList.toggle('d-none', validBandpass && validPercent);
        preprocessingConfirmed = validBandpass && validPercent;
        updateSteps();
    });
    document.getElementById('preprocessing-skip').addEventListener('click', () => {
        ['apply-notch', 'apply-bandpass', 'apply-channel-qc'].forEach(id => {
            document.getElementById(id).checked = false;
        });
        document.getElementById('preprocessing-error').classList.add('d-none');
        preprocessingConfirmed = true;
        updateSteps();
    });
    updateSteps();
});

// ---------- Drag & drop + file selection UI ----------

document.addEventListener('DOMContentLoaded', () => {
    // Cache references to key DOM elements related to file upload
    const dropZone = document.getElementById('drop-zone');                 // The clickable + droppable area
    const fileInput = document.getElementById('edf-files');               // Hidden <input type="file">
    const fileList = document.getElementById('file-list');                // Container for showing selected file badges
    const fileListContainer = document.getElementById('file-list-container'); // Wrapper around file list (toggle visibility)

    // 1. When user clicks the drop zone, open the system file picker
    dropZone.addEventListener('click', () => fileInput.click());

    // 2. Prevent default browser behavior for drag events (e.g., opening files in the browser)
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();   // Prevent default behavior (e.g., open file in new tab)
            e.stopPropagation(); // Stop event from bubbling up further
        }, false);
    });

    // 3. Visual feedback: add a CSS class when files are dragged over the zone
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    // Remove the visual class when drag leaves or files are dropped
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    // 4. Handle the actual drop event (core of drag-and-drop upload)
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;  // DataTransfer object from the drop event
        const files = dt.files;     // FileList containing dragged files

        if (files.length > 0) {
            // Sync the dragged files with the real hidden file input
            fileInput.files = files;

            // Manually fire a 'change' event so any listeners on the input update the UI
            const event = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(event);
        }
    });

    // 5. When files are selected (either via click or drag & drop), update the UI list
    fileInput.addEventListener('change', function() {
        fileList.innerHTML = ''; // Clear previous badges

        if (this.files && this.files.length > 0) {
            // Show the container that lists selected files
            fileListContainer.classList.remove('d-none');

            // For each selected file, create a small "badge" with an icon + filename
            Array.from(this.files).forEach(file => {
                const div = document.createElement('div');
                div.className = 'file-badge';
                div.innerHTML = `<i class="bi bi-file-earmark-binary me-2"></i>${file.name}`;
                fileList.appendChild(div);
            });

            console.log("Files ready to be sent:", this.files);
        } else {
            // If no files remain, hide the list container
            fileListContainer.classList.add('d-none');
        }
    });
});


// ---------- Theme initialization (dark / light) ----------

// Initialize theme and UI styling once DOM is ready
document.addEventListener('DOMContentLoaded', async () => {

    // Map of button IDs grouped by their intended "style" (primary vs secondary)
    // This will be used to toggle between filled vs outline style based on theme.
    const buttonGroups = {
        primary: [
            'predictButton'
        ],
        secondary: [
            'resetButton',
            'next',
            'prev',
            'nextHypno',
            'prevHypno',
        ]
    };

    // Detect OS/browser preference for dark mode using matchMedia
    isDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    console.log(isDarkMode ? 'Dark mode enabled' : 'Light mode enabled');

    // Manually set background colors of main cards based on theme
    document.getElementById('card-uploader').style.backgroundColor    = isDarkMode ? '#272a2e' : '#f7f7f7';
    document.getElementById('channels-card').style.backgroundColor    = isDarkMode ? '#272a2e' : '#f7f7f7';
    document.getElementById('algorithms-select').style.backgroundColor = isDarkMode ? '#272a2e' : '#f7f7f7';

    // Adjust color of info links for each model/ensemble
    document.getElementById('usleep-link').style.color      = isDarkMode ? '#ffffff' : '#000000';
    document.getElementById('deepresnet-link').style.color  = isDarkMode ? '#ffffff' : '#000000';
    document.getElementById('sleeptransformer-link').style.color = isDarkMode ? '#ffffff' : '#000000';
    document.getElementById('ensemble-link').style.color    = isDarkMode ? '#ffffff' : '#000000';

    // Swap logo based on theme and inform Bootstrap via data-bs-theme
    document.getElementById('logo').src = isDarkMode
        ? '../static/images/sleepyland_logo_dark.png'
        : '../static/images/sleepyland_logo_light.png';
    document.body.dataset.bsTheme = isDarkMode ? 'dark' : 'light';

    // Toggle Bootstrap button classes based on theme:
    //  - In light mode: use solid btn-primary / btn-secondary
    //  - In dark mode: use outline variants btn-outline-primary / btn-outline-secondary
    Object.entries(buttonGroups).forEach(([style, buttons]) => {
        const lightStyle = `btn-${style}`;            // e.g. 'btn-primary'
        const darkStyle = `btn-outline-${style}`;     // e.g. 'btn-outline-primary'
        buttons.forEach(buttonId => {
            const button = document.getElementById(buttonId);
            if (!button) return; // in case the element does not exist

            // If not dark mode: use filled button; if dark mode: use outline
            button.classList.toggle(lightStyle, !isDarkMode);
            button.classList.toggle(darkStyle, isDarkMode);
        });
    });

});


// ---------- Ensemble checkbox logic + Resolution slider ----------

const RESOLUTION_VALUES = [1, 2, 5, 10, 30];

document.addEventListener('DOMContentLoaded', () => {
    const ensembleCheckbox = document.getElementById('ensemble'); // The ensemble model checkbox
    const ensembleMethodSelect = document.getElementById('ensemble-method');
    const algorithms = document.querySelectorAll('#algorithms-select input[type="checkbox"]');
    const resolutionSlider = document.getElementById('resolution-slider');
    const resolutionDisplay = document.getElementById('resolution-value');

    // Enable/disable resolution slider: only active when USleep is the sole base model selected
    const updateResolutionSlider = () => {
        const baseChecked = Array.from(algorithms).filter(cb => cb.checked && cb.id !== 'ensemble');
        const onlyUsleep = baseChecked.length === 1 && baseChecked[0].id === 'usleep';

        resolutionSlider.disabled = !onlyUsleep;

        if (!onlyUsleep) {
            // Lock back to 30s (index 4) when slider is not applicable
            resolutionSlider.value = 4;
            resolutionDisplay.textContent = 30;
        }
    };

    // Update displayed value when slider moves
    resolutionSlider.addEventListener('input', () => {
        resolutionDisplay.textContent = RESOLUTION_VALUES[parseInt(resolutionSlider.value)];
    });

    // Function that enables/disables the ensemble checkbox depending on how many models are selected
    const updateEnsembleCheckboxState = () => {
        // Count all checked checkboxes within the algorithms section
        let checkedCount = Array.from(algorithms).filter(algorithm => algorithm.checked).length;

        // If ensemble is itself checked, subtract 1 so you only count base models
        if (ensembleCheckbox.checked) {
            checkedCount = Array.from(algorithms).filter(algorithm => algorithm.checked).length - 1;
        }

        // Ensemble needs at least 2 base models to make sense
        ensembleCheckbox.disabled = checkedCount < 2;

        // If we disable it (because too few models), also uncheck it
        if (ensembleCheckbox.disabled) {
            ensembleCheckbox.checked = false;
        }

        // The method is only relevant when a valid ensemble is selected.
        ensembleMethodSelect.disabled = ensembleCheckbox.disabled || !ensembleCheckbox.checked;
    };

    // Watch all algorithm checkboxes for changes to recalculate ensemble eligibility and slider state
    algorithms.forEach(algorithm => {
        algorithm.addEventListener('change', () => {
            updateEnsembleCheckboxState();
            updateResolutionSlider();
        });
    });

    // When ensemble changes, just recalc to ensure consistent state
    ensembleCheckbox.addEventListener('change', () => {
        updateEnsembleCheckboxState();
    });

    // Initialize both states on page load
    updateEnsembleCheckboxState();
    updateResolutionSlider();
});


// ---------- Handling server response after prediction ----------

// Handles the response of /process from the backend
const handleProcessingResponsePredictOne = (response, data, checkedAlgorithms) => {
    const status = document.getElementById('status');         // Text status area under Hypnograms
    const statusHypno = document.getElementById('statusHypno'); // Text status area under Hypnodensity

    if (response.ok) {
        // Clear any error messages
        status.textContent = '';
        statusHypno.textContent = '';

        // Create empty cards for each expanded model+lead tag in both tabs
        createCards(expandedModelsGlobal);

        // Load total number of slides and then the data for the first slide
        loadTotalSlides(true);
    } else {
        // If response is not OK, show error text in both tabs
        status.textContent = data.error || 'An error occurred.';
        statusHypno.textContent = data.error || 'An error occurred.';
    }
};


// ---------- Creating result cards for each algorithm ----------

// Groups expanded model tags (e.g. ['usleep_eeg','usleep_eog','ensemble']) by base model.
// Returns an ordered Map: { 'usleep' => ['eeg','eog'], 'ensemble' => [] }
const groupModelsByBase = (expandedModels) => {
    const groups = new Map();
    for (const tag of expandedModels) {
        if (tag === 'ensemble') {
            if (!groups.has('ensemble')) groups.set('ensemble', []);
            groups.get('ensemble').push('all');
        } else {
            const idx = tag.indexOf('_');
            const base = tag.substring(0, idx);
            const lead = tag.substring(idx + 1);
            if (!groups.has(base)) groups.set(base, []);
            groups.get(base).push(lead);
        }
    }
    return groups;
};

const baseModelLabel = (base) => {
    if (base === 'usleep') return 'USleep';
    if (base === 'deepresnet') return 'DeepResNet';
    if (base === 'sleeptransformer') return 'SleepTransformer';
    if (base === 'ensemble') return 'Ensemble';
    return base;
};

// Creates one grouped card per base model with Bootstrap tabs for each lead type.
const createGroupCard = (base, leads, container, type) => {
    const card = document.createElement('div');
    card.classList.add('card', 'm-3', 'shadow-sm', 'rounded');

    // --- Header ---
    const cardHeader = document.createElement('div');
    cardHeader.classList.add('card-header', 'd-flex', 'justify-content-between', 'align-items-center');

    const headerText = document.createElement('span');
    headerText.textContent = baseModelLabel(base);

    const arrowIcon = document.createElement('span');
    arrowIcon.classList.add('arrow-icon');
    arrowIcon.textContent = '▲';

    cardHeader.appendChild(headerText);
    cardHeader.appendChild(arrowIcon);
    cardHeader.addEventListener('click', () => {
        const cardBody = card.querySelector('.card-body');
        const isHidden = cardBody.style.display === 'none';
        cardBody.style.display = isHidden ? 'block' : 'none';
        arrowIcon.textContent = isHidden ? '▲' : '▼';
    });
    card.appendChild(cardHeader);

    // --- Body ---
    const cardBody = document.createElement('div');
    cardBody.classList.add('card-body', 'p-2');

    const chartHeight = '450px';
    const containerStyle = `border: 2px solid #636161; border-radius: 12px; height: ${chartHeight}; width: 100%;`;
    const bgColor = isDarkMode ? '#272a2e' : '#f8f9fa';

    if (leads.length <= 1) {
        // Single chart, no tabs needed
        const tag = base === 'ensemble' && leads[0] === 'all' ? 'ensemble' : `${base}_${leads[0]}`;
        cardBody.style.height = '500px';
        const chartDiv = document.createElement('div');
        chartDiv.id = `${type}-container-${tag}`;
        chartDiv.style.cssText = containerStyle;
        chartDiv.style.backgroundColor = bgColor;
        cardBody.appendChild(chartDiv);
    } else {
        // Multiple leads → Bootstrap pill tabs
        cardBody.style.height = '530px';

        const navId = `nav-${type}-${base}`;

        // Tab nav (pills style for a cleaner look)
        const tabNav = document.createElement('ul');
        tabNav.classList.add('nav', 'nav-pills', 'mb-2', 'gap-1');
        tabNav.id = navId;
        tabNav.setAttribute('role', 'tablist');

        const tabContent = document.createElement('div');
        tabContent.classList.add('tab-content');

        leads.forEach((lead, idx) => {
            const tag = base === 'ensemble' && lead === 'all' ? 'ensemble' : `${base}_${lead}`;
            const paneId = `pane-${type}-${tag}`;

            // Nav pill
            const navItem = document.createElement('li');
            navItem.classList.add('nav-item');
            navItem.setAttribute('role', 'presentation');

            const navBtn = document.createElement('button');
            navBtn.classList.add('nav-link', 'btn-sm', 'px-3', 'py-1');
            if (idx === 0) navBtn.classList.add('active');
            navBtn.id = `btn-${type}-${tag}`;
            navBtn.setAttribute('data-bs-toggle', 'pill');
            navBtn.setAttribute('data-bs-target', `#${paneId}`);
            navBtn.setAttribute('type', 'button');
            navBtn.setAttribute('role', 'tab');
            navBtn.textContent = lead.toUpperCase();
            navItem.appendChild(navBtn);
            tabNav.appendChild(navItem);

            // Tab pane
            const tabPane = document.createElement('div');
            tabPane.classList.add('tab-pane', 'fade');
            if (idx === 0) tabPane.classList.add('show', 'active');
            tabPane.id = paneId;
            tabPane.setAttribute('role', 'tabpanel');
            tabPane.style.height = chartHeight;

            const chartDiv = document.createElement('div');
            chartDiv.id = `${type}-container-${tag}`;
            chartDiv.style.cssText = containerStyle;
            chartDiv.style.backgroundColor = bgColor;
            tabPane.appendChild(chartDiv);
            tabContent.appendChild(tabPane);

            // Resize Plotly plot when tab becomes visible (hidden tabs have 0 dimensions)
            navBtn.addEventListener('shown.bs.tab', () => {
                const el = document.getElementById(`${type}-container-${tag}`);
                if (el) Plotly.Plots.resize(el);
            });
        });

        cardBody.appendChild(tabNav);
        cardBody.appendChild(tabContent);
    }

    card.appendChild(cardBody);
    container.appendChild(card);
};

// Dynamically creates grouped cards for hypnograms and hypnodensity tabs.
const createCards = (expandedModels) => {
    const cardContainer = document.getElementById('card-container');
    const cardContainerHypno = document.getElementById('card-container-hypno');
    cardContainer.innerHTML = '';
    cardContainerHypno.innerHTML = '';

    // Hide empty-state placeholders when results are being loaded
    document.getElementById('hypnogram-placeholder').classList.add('d-none');
    document.getElementById('hypnodensity-placeholder').classList.add('d-none');

    const groups = groupModelsByBase(expandedModels);
    for (const [base, leads] of groups) {
        createGroupCard(base, leads, cardContainer, 'hypnogram_combined');
        createGroupCard(base, leads, cardContainerHypno, 'hypnodensity');
    }
};


// ---------- Loading the total number of slides ----------

const loadTotalSlides = (is_predict) => {
    fetch('../static/total_slides.json')
        .then(response => response.json())
        .then(data => {
            // total_slides.json is expected to have a field "total_slides"
            totalSlides = data.total_slides;

            // Load initial hypnogram and hypnodensity plots for the first slide index
            loadData(currentIndex, 'hypnogram_combined');
            loadData(currentIndex, 'hypnodensity');

            if (!is_predict) {
                loadPerformanceData(currentIndex);
            }
        })
        .catch(error => console.error('Error loading total slides:', error));
};


// ---------- Loading JSON data and plotting with Plotly ----------

// Load model output JSONs for a given slide index and type ('hypnogram_combined' or 'hypnodensity')
const loadData = (slideIndex, type) => {
    // Use expanded model+lead tags (e.g. 'usleep_eeg') set when prediction was launched,
    // so paths match exactly what graph_service.py wrote to /app/static/.
    const modelFiles = expandedModelsGlobal.map(algorithm => ({
        algorithm,
        file: `../static/${algorithm}/${type}_${algorithm}_${slideIndex}.json`
    }));

    // Decide which part of the DOM to target based on the type:
    const resultsDiv = document.getElementById(
        type === 'hypnogram_combined' ? 'results' : 'resultsHypno'
    );
    const buttonsDiv = document.getElementById(
        type === 'hypnogram_combined' ? 'nextPrevButtonDiv' : 'nextPrevButtonDivHypno'
    );
    const cardContainer = document.getElementById(
        type === 'hypnogram_combined' ? 'card-container' : 'card-container-hypno'
    );

    // Keep card-container, nav buttons and the empty-state placeholder; remove anything else
    const placeholder = document.getElementById(
        type === 'hypnogram_combined' ? 'hypnogram-placeholder' : 'hypnodensity-placeholder'
    );
    Array.from(resultsDiv.children).forEach(child => {
        if (child !== buttonsDiv && child !== cardContainer && child !== placeholder) {
            resultsDiv.removeChild(child);
        }
    });

    // For each expanded model+lead tag, fetch and render its JSON plot
    modelFiles.forEach(({ algorithm, file }) => {
        fetch(file)
            .then(response => response.json())
            .then(data => {
                // The JSON file is assumed to contain a JSON-stringified Plotly spec,
                // so parse it again here to get the actual object.
                const plotData = JSON.parse(data);

                // Apply the shared high-contrast plot palette in either UI theme
                applyThemeToPlot(plotData);

                // Render Plotly chart into the container that was created in createCards
                Plotly.newPlot(
                    `${type}-container-${algorithm}`,
                    plotData.data,
                    plotData.layout,
                    { responsive: true }
                );

                // Ensure styling of the container is correct
                const container = document.getElementById(`${type}-container-${algorithm}`);
                container.style.borderRadius = '15px';
                container.style.overflow = 'hidden';

                // Show navigation buttons if they were hidden
                if (buttonsDiv.style.display === 'none') {
                    buttonsDiv.style.display = 'block';
                }

                // Use ResizeObserver to auto-resize the plot when the container size changes
                const resizeObserver = new ResizeObserver(() => {
                    Plotly.Plots.resize(container);
                });
                resizeObserver.observe(container);
            })
            .catch(error => console.error(`Error loading ${type} from ${file}:`, error));
    });
};


// ---------- Apply theme to Plotly layouts ----------

// Keep plot colors consistent in the light and dark UI themes.
const applyThemeToPlot = (plotData) => {
    plotData.layout.plot_bgcolor = '#0C1729';
    plotData.layout.paper_bgcolor = '#0C1729';
    plotData.layout.font = { color: '#A9BAD3' };
    for (const axis of ['xaxis', 'yaxis']) {
        plotData.layout[axis] ??= {};
        plotData.layout[axis].tickfont = { color: '#A9BAD3' };
        plotData.layout[axis].gridcolor = 'rgba(54,73,101,0.45)';
        plotData.layout[axis].linecolor = '#364965';
        plotData.layout[axis].showline = true;
        plotData.layout[axis].mirror = true;
    }
    plotData.layout.autosize = true;
};


// ---------- Slide navigation (prev/next) ----------

// Navigate between slides for hypnograms
document.getElementById('prev').addEventListener('click', () => {
    // Decrement slide index and wrap around using modulo arithmetic
    currentSlide = (currentSlide - 1 + totalSlides) % totalSlides;
    loadData(currentSlide, 'hypnogram_combined');
});

document.getElementById('next').addEventListener('click', () => {
    // Increment slide index and wrap around
    currentSlide = (currentSlide + 1) % totalSlides;
    loadData(currentSlide, 'hypnogram_combined');
});

// Navigate between slides for hypnodensity plots
document.getElementById('prevHypno').addEventListener('click', () => {
    currentSlide = (currentSlide - 1 + totalSlides) % totalSlides;
    loadData(currentSlide, 'hypnodensity');
});

document.getElementById('nextHypno').addEventListener('click', () => {
    currentSlide = (currentSlide + 1) % totalSlides;
    loadData(currentSlide, 'hypnodensity');
});


// ---------- Reset button (backend cleanup) ----------

document.getElementById('resetButton').addEventListener('click', async () => {
    // Ask backend to clear ./input via /reset-directories
    const response = await fetch('/reset-directories', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    });

    if (response.ok) {
        // On success, you parse JSON but don't show it.
        // You could add a toast or console log if desired.
        const data = await response.json();
        console.log(data);
    } else {
        const errorData = await response.json();
        alert(`Error: ${errorData.detail}`);
    }
});


// ---------- Stop button (stops all containers) ----------

document.getElementById('stopAppButton').addEventListener('click', async () => {
    const confirmed = await showConfirmModal(
        'This stops all SLEEPYLAND containers. Any prediction still running will be cancelled. ' +
        'Your input and output files are kept. To use SLEEPYLAND again, start it from the launcher.',
        { title: 'Stop SLEEPYLAND?', confirmLabel: 'Stop', icon: 'bi-power' }
    );
    if (!confirmed) return;

    const button = document.getElementById('stopAppButton');
    button.disabled = true;
    button.textContent = 'Stopping…';

    try {
        const response = await fetch('/quit', { method: 'POST' });
        const data = await response.json();
        if (data.unreachable && data.unreachable.length) {
            console.warn('Services that did not answer (may already be stopped):', data.unreachable);
        }
    } catch (error) {
        // The GUI may exit before the reply arrives; the containers are stopping either way.
        console.warn('No reply from /quit:', error);
    }

    // Replace the page: nothing on it works once the containers are gone.
    const overlay = document.createElement('div');
    overlay.className = 'app-stopped-overlay';
    overlay.innerHTML =
        '<div><i class="bi bi-power fs-1 d-block mb-3"></i>' +
        '<h4>SLEEPYLAND has been stopped</h4>' +
        '<p class="text-muted mb-0">All containers are stopped. You can close this tab.</p></div>';
    document.body.appendChild(overlay);
});


// ---------- Utility: get selected channels ----------

// Returns array of selected modality values, e.g. ["eeg", "eog"]
function getCheckedChannels() {
    // Select all checked modality checkboxes
    const checkboxes = document.querySelectorAll('input[name="select-channels-proprietary"]:checked');
    // Map NodeList to array of values
    return Array.from(checkboxes).map(checkbox => checkbox.value);
}


// ---------- Predict button: full front-end workflow ----------

document.getElementById('predictButton').addEventListener('click', async () => {
    const button = document.getElementById('predictButton');
    const status = document.getElementById('status');
    const statusHypno = document.getElementById('statusHypno');

    // Validation badges under "Output Folder and Available Modalities" and "Sleep Staging Models"
    const validationBadgeTypeChannels = document.getElementById("validationBadgeTypeChannels");
    const validationBadgeAlgorithms = document.getElementById("validationBadgeAlgorithms");
    const validationBadgeEdf = document.getElementById("validationBadgeEdf");
    const validationBadgeFolderName = document.getElementById("validationBadgeOutputFolder");
    const reviewSoftwareSelect = document.getElementById('review-software');

    const formData = new FormData();
    const files = document.getElementById('edf-files').files;

    // Append all selected EDF files to FormData under the same field name 'edf-files'
    for (let i = 0; i < files.length; i++) {
        formData.append('edf-files', files[i]);
    }

    // Append folder name (used as output folder in backend)
    const folderName = document.getElementById('folderName').value;
    formData.append('folderName', folderName);

    // Collect all checked model checkboxes within the algorithms section
    const checkboxes = document.querySelectorAll('#algorithms-select input[type="checkbox"]:checked');

    // Validation flags
    const isAnyAlgorithmChecked = Array.from(checkboxes).some(checkbox => checkbox.checked);
    const isAnyChannelSelected = Array.from(getCheckedChannels()).length > 0;
    const isAnyFileSelected = files.length > 0;
    const isFolderNameValid = folderName.length > 0;
    const isReviewSoftwareSelected = reviewSoftwareSelect.value !== '';
    const low = Number(document.getElementById('bandpass-low').value);
    const high = Number(document.getElementById('bandpass-high').value);
    const bandpassValid = !document.getElementById('apply-bandpass').checked || (low > 0 && high > low);
    const maxBadPercent = Number(document.getElementById('max-bad-epoch-percent').value);
    const channelQcValid = !document.getElementById('apply-channel-qc').checked ||
        (maxBadPercent >= 0 && maxBadPercent <= 100 && document.getElementById('max-bad-epoch-percent').value !== '');
    const preprocessingError = document.getElementById('preprocessing-error');
    preprocessingError.textContent = !bandpassValid ? 'Enter positive bandpass cutoffs with the high cutoff above the low cutoff.' :
        'Enter a maximum flagged epoch percentage between 0 and 100.';
    preprocessingError.classList.toggle('d-none', bandpassValid && channelQcValid);

    // Toggle visibility of validation badges based on each condition
    validationBadgeTypeChannels.classList.toggle("d-none", isAnyChannelSelected);
    validationBadgeAlgorithms.classList.toggle("d-none", isAnyAlgorithmChecked);
    validationBadgeEdf.classList.toggle("d-none", isAnyFileSelected);
    validationBadgeFolderName.classList.toggle("d-none", isFolderNameValid);
    reviewSoftwareSelect.classList.toggle('is-invalid', !isReviewSoftwareSelected);

    // Single shared progress bar (above tab content, always visible during prediction)
    const progressContainer = document.getElementById('global-progress-container');
    const progressBar       = document.getElementById('global-progress-bar');
    const statusLabel       = document.getElementById('global-status-label');
    const statusPercent     = document.getElementById('global-status-percent');

    // Show progress bar and reset appearance
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    progressBar.classList.add('progress-bar-animated');
    progressBar.classList.replace('bg-success', 'bg-primary');
    statusLabel.textContent = 'Starting…';
    statusPercent.textContent = '0%';

    // Only proceed if all basic validations pass
    if (isAnyChannelSelected && isAnyAlgorithmChecked && isAnyFileSelected && isFolderNameValid && isReviewSoftwareSelected && bandpassValid && channelQcValid) {
        formData.append('apply_notch', document.getElementById('apply-notch').checked);
        formData.append('notch_freq', document.getElementById('notch-freq').value);
        formData.append('apply_bandpass', document.getElementById('apply-bandpass').checked);
        formData.append('bandpass_low', low);
        formData.append('bandpass_high', high);
        formData.append('apply_quality_control', document.getElementById('apply-qc').checked);
        formData.append('apply_channel_qc', document.getElementById('apply-channel-qc').checked);
        formData.append('max_bad_epoch_fraction', maxBadPercent / 100);

        // Append selected channels as a single field (backend splits comma-separated string)
        formData.append('channels', getCheckedChannels());

        // Build a list of algorithm IDs (e.g., ["usleep", "deepresnet", "ensemble"])
        const checkedAlgorithms = [...document.querySelectorAll('#algorithms-select input[type="checkbox"]:checked')]
            .map(checkbox => checkbox.id);

        // Append comma-separated models list for the backend
        formData.append('models', checkedAlgorithms.join(','));

        // Append selected subject type (adult / child) for model weight selection
        const modelSet = document.querySelector('input[name="model_set"]:checked').value;
        formData.append('model_set', modelSet);
        formData.append('review_software', reviewSoftwareSelect.value);

        // Append resolution (seconds): read from slider mapping, default 30 if slider disabled
        const sliderEl = document.getElementById('resolution-slider');
        const resolution = RESOLUTION_VALUES[parseInt(sliderEl.value)];
        formData.append('resolution', resolution);

        // This value is used only when the Ensemble checkbox is selected.
        formData.append('ensemble_method', document.getElementById('ensemble-method').value);

        // Mirror the backend expansion: pair each base model with each selected lead type
        // so frontend paths match exactly what graph_service.py writes to /app/static/.
        const checkedChannels = getCheckedChannels(); // e.g. ['eeg', 'eog']
        expandedModelsGlobal = [];
        for (const m of checkedAlgorithms) {
            if (m === 'ensemble') {
                expandedModelsGlobal.push('ensemble');
            } else {
                for (const channel of checkedChannels) {
                    // run.py writes EMG predictions under the internal "chin" lead type.
                    const lt = channel === 'emg' ? 'chin' : channel;
                    expandedModelsGlobal.push(`${m}_${lt}`);
                }
            }
        }

        // Disable Predict button and show loading state
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Predicting…';

        // Hide navigation buttons while processing, force with !important
        document.getElementById('nextPrevButtonDiv').style.setProperty('display', 'none', 'important');
        document.getElementById('nextPrevButtonDivHypno').style.setProperty('display', 'none', 'important');

        // Clear cards, hide title and placeholders — progress bar will fill the space
        document.getElementById('card-container').innerHTML = '';
        document.getElementById('card-container-hypno').innerHTML = '';
        document.getElementById('hypnogram-placeholder').classList.add('d-none');
        document.getElementById('hypnodensity-placeholder').classList.add('d-none');
        setStudyTitle(null);

        // Programmatically switch to the Hypnograms tab to show progress/results
        new bootstrap.Tab(document.getElementById('hypnogram-tab')).show();
        // Scroll to top of page for better UX
        window.scrollTo(0, 0);

        status.textContent = '';

        // Simulated creep: advances +1% every 2.5 s up to 75% while waiting for inference.
        // Real server values (from polling) override it via Math.max, so the bar never
        // goes backwards and real updates (85%, 90%, 95%) take over cleanly.
        let simulatedPercent = 0;
        const creepInterval = setInterval(() => {
            if (simulatedPercent < 75) simulatedPercent += 1;
        }, 2500);

        // Start polling the backend /progress/<folderName> every 1 second
        const pollingInterval = setInterval(async () => {
            try {
                const res = await fetch(`/progress/${folderName}`);
                const data = await res.json();

                if (data.percent !== undefined) {
                    const displayed = Math.max(data.percent, simulatedPercent);
                    const message   = data.message || 'Processing…';

                    progressBar.style.width = displayed + '%';
                    statusPercent.textContent = displayed + '%';
                    statusLabel.textContent = message;
                }
            } catch (e) {
                console.error("Polling error:", e);
            }
        }, 1000);

        try {
            const response = await fetch('/process', { method: 'POST', body: formData });

            let data = {};
            try {
                data = await response.json();
            } catch (e) {
                data = { error: 'Server did not return valid JSON.' };
            }

            // Stop both intervals and hide the progress bar
            clearInterval(creepInterval);
            clearInterval(pollingInterval);
            progressContainer.style.display = 'none';

            // Show project name in the tab banner only on success
            if (response.ok) {
                setStudyTitle(folderName);
                if (Array.isArray(data.expanded_models)) expandedModelsGlobal = data.expanded_models;
            }

            handleProcessingResponsePredictOne(response, data, checkedAlgorithms);
        } catch (error) {
            status.textContent = 'An error occurred: ' + error;
        } finally {
            // Restore button to original state
            button.disabled = false;
            button.innerHTML = 'Predict';
        }
    }
});


// ---------- Shared helpers ----------

// Promisified Bootstrap confirmation modal (replaces browser confirm())
const showConfirmModal = (message, options = {}) => new Promise((resolve) => {
    const title = options.title || 'Confirm deletion';
    const confirmLabel = options.confirmLabel || 'Delete';
    document.getElementById('confirmModalBody').textContent = message;
    document.getElementById('confirmModalLabel').textContent = title;
    const modalEl = document.getElementById('confirmModal');
    const modal   = new bootstrap.Modal(modalEl);
    const okBtn   = document.getElementById('confirmModalOk');

    okBtn.classList.toggle('btn-danger', options.danger !== false);
    okBtn.classList.toggle('btn-primary', options.danger === false);
    const icon = options.icon || (options.danger === false ? 'bi-arrow-repeat' : 'bi-trash');
    okBtn.innerHTML = `<i class="bi ${icon} me-1"></i>${confirmLabel}`;

    let confirmed = false;

    const onOk = () => { confirmed = true; modal.hide(); };
    okBtn.addEventListener('click', onOk, { once: true });
    modalEl.addEventListener('hidden.bs.modal', () => resolve(confirmed), { once: true });

    modal.show();
});

// Show/hide the project name inside the Hypnograms and Hypnodensity tab-panes
const setStudyTitle = (name) => {
    ['hypnogram', 'hypnodensity'].forEach(type => {
        const titleDiv = document.getElementById(`${type}-study-title`);
        const nameSpan = document.getElementById(`${type}-study-name`);
        if (name) {
            nameSpan.textContent = name;
            titleDiv.classList.remove('d-none');
        } else {
            titleDiv.classList.add('d-none');
        }
    });
};


// ---------- History tab ----------

const deleteStudy = async (folderName, btn) => {
    const confirmed = await showConfirmModal(`Delete "${folderName}"? This cannot be undone.`);
    if (!confirmed) return;

    // Find the parent row by its unique class combination (not by data-folder,
    // since the button itself also carries that attribute)
    const row = btn.closest('.border.rounded');

    // Animate row out immediately — no need to wait for the server response
    row.style.transition = 'opacity 0.25s';
    row.style.opacity = '0';

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span>';

    try {
        const res  = await fetch(`/delete/${folderName}`, { method: 'POST' });
        const data = await res.json();

        if (!res.ok) {
            // Restore row visibility on error
            row.style.opacity = '1';
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-trash"></i>';
            showConfirmModal(data.error || 'Failed to delete study.');
            return;
        }

        setTimeout(() => {
            row.remove();
            const listEl = document.getElementById('history-list');
            if (!listEl.querySelector('.border.rounded')) {
                listEl.innerHTML = '<p class="text-muted small">No previous runs found.</p>';
            }
        }, 250);

    } catch (e) {
        row.style.opacity = '1';
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-trash"></i>';
        console.error('deleteStudy error:', e);
    }
};

const loadHistory = async () => {
    const listEl = document.getElementById('history-list');
    listEl.innerHTML = '<p class="text-muted small">Loading…</p>';

    try {
        const res = await fetch('/get_studies');
        const studies = await res.json();

        if (!studies.length) {
            listEl.innerHTML = '<p class="text-muted small">No previous runs found.</p>';
            return;
        }

        // Sort alphabetically descending so the most recent (by name) appears first
        studies.sort((a, b) => b.localeCompare(a));

        listEl.innerHTML = '';
        studies.forEach(folder => {
            const row = document.createElement('div');
            row.className = 'border rounded p-3 mb-2 d-flex justify-content-between align-items-center';
            row.dataset.folder = folder;
            row.innerHTML = `
                <span class="fw-semibold">${folder}</span>
                <div class="d-flex gap-2">
                    <button class="btn btn-sm btn-outline-primary load-study-btn" data-folder="${folder}">
                        <i class="bi bi-graph-up me-1"></i>Load
                    </button>
                    <button class="btn btn-sm btn-outline-danger delete-study-btn" data-folder="${folder}" title="Delete">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>`;
            listEl.appendChild(row);
        });

        document.querySelectorAll('.load-study-btn').forEach(btn => {
            btn.addEventListener('click', () => loadStudy(btn.dataset.folder, btn));
        });

        document.querySelectorAll('.delete-study-btn').forEach(btn => {
            btn.addEventListener('click', () => deleteStudy(btn.dataset.folder, btn));
        });

    } catch (e) {
        listEl.innerHTML = '<p class="text-danger small">Failed to load history.</p>';
        console.error('History load error:', e);
    }
};

const loadStudy = async (folderName, btn) => {
    const progressContainer = document.getElementById('global-progress-container');
    const progressBar       = document.getElementById('global-progress-bar');
    const statusLabel       = document.getElementById('global-status-label');
    const statusPercent     = document.getElementById('global-status-percent');

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status"></span>Loading…';

    progressContainer.style.display = 'block';
    progressBar.classList.add('progress-bar-animated');
    progressBar.classList.replace('bg-success', 'bg-primary');
    progressBar.style.width = '30%';
    statusLabel.textContent = 'Generating charts…';
    statusPercent.textContent = '30%';

    try {
        const res  = await fetch(`/load/${folderName}`, { method: 'POST' });
        const data = await res.json();

        if (!res.ok) {
            alert(data.error || 'Failed to load study.');
            return;
        }

        progressBar.style.width = '100%';
        statusPercent.textContent = '100%';
        statusLabel.textContent = 'Done';

        expandedModelsGlobal = data.expanded_models;
        setStudyTitle(folderName);

        // Clear existing cards and rebuild for the loaded study
        document.getElementById('card-container').innerHTML = '';
        document.getElementById('card-container-hypno').innerHTML = '';
        document.getElementById('hypnogram-placeholder').classList.remove('d-none');
        document.getElementById('hypnodensity-placeholder').classList.remove('d-none');
        document.getElementById('nextPrevButtonDiv').style.setProperty('display', 'none', 'important');
        document.getElementById('nextPrevButtonDivHypno').style.setProperty('display', 'none', 'important');

        createCards(expandedModelsGlobal);
        loadTotalSlides(true);

        // Switch to Hypnograms tab to show results
        new bootstrap.Tab(document.getElementById('hypnogram-tab')).show();
        window.scrollTo(0, 0);

    } catch (e) {
        alert('An error occurred: ' + e);
        console.error('loadStudy error:', e);
    } finally {
        progressContainer.style.display = 'none';
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-graph-up me-1"></i>Load';
    }
};

document.addEventListener('DOMContentLoaded', () => {
    // Auto-load history list when the History tab is first activated
    document.getElementById('history-tab').addEventListener('shown.bs.tab', loadHistory);

    // Refresh button inside the History tab
    document.getElementById('refreshHistoryBtn').addEventListener('click', loadHistory);
});
