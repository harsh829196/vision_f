/**
 * EndoVision AI — Advanced Endoscopic GI Polyp & Lesion Segmentation Frontend
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    currentSample: 'sessile',
    currentMode: 'split',
    currentColor: '#10b981',
    currentOpacity: 0.55,
    currentThreshold: 0.45,
    lastAnalysis: null,
    isAnalyzing: false,
    zoomLevel: 1.0,
    isStreaming: false,
    streamInterval: null,
    streamIndex: 0
  };

  // DOM Elements
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const btnBrowse = document.getElementById('btn-browse');
  const scanline = document.getElementById('scanline');
  const stageLoader = document.getElementById('stage-loader');

  // Split Slider Elements
  const splitWrapper = document.getElementById('split-wrapper');
  const splitTopWrapper = document.getElementById('split-top-wrapper');
  const splitDivider = document.getElementById('split-divider');
  const splitRawImg = document.getElementById('split-raw-img');
  const splitOverlayImg = document.getElementById('split-overlay-img');

  // Views
  const viewSplit = document.getElementById('view-split');
  const viewSingle = document.getElementById('view-single');
  const viewQuad = document.getElementById('view-quad');
  const singleDisplayImg = document.getElementById('single-display-img');
  const singleLayerTag = document.getElementById('single-layer-tag');
  const singleZoomContainer = document.getElementById('single-zoom-container');

  // Quad Views
  const quadRaw = document.getElementById('quad-raw');
  const quadOverlay = document.getElementById('quad-overlay');
  const quadHeatmap = document.getElementById('quad-heatmap');
  const quadMask = document.getElementById('quad-mask');

  // Tabs & Controls
  const modeTabs = document.querySelectorAll('.mode-tab');
  const sampleCards = document.querySelectorAll('.sample-card');
  const colorSwatches = document.querySelectorAll('.color-swatch');
  const opacitySlider = document.getElementById('opacity-slider');
  const opacityVal = document.getElementById('opacity-val');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdVal = document.getElementById('threshold-val');
  const btnReanalyze = document.getElementById('btn-reanalyze');
  const colorHexLabel = document.getElementById('color-hex-label');

  // Dashboard Telemetry Elements
  const diagBanner = document.getElementById('diag-banner');
  const bannerIcon = document.getElementById('banner-icon');
  const detectionBadge = document.getElementById('detection-badge');
  const parisBadge = document.getElementById('paris-badge');
  const morphologyText = document.getElementById('morphology-text');
  const confVal = document.getElementById('conf-val');
  const confBarFill = document.getElementById('conf-bar-fill');

  const metricAreaPct = document.getElementById('metric-area-pct');
  const metricPixels = document.getElementById('metric-pixels');
  const metricTotalPixels = document.getElementById('metric-total-pixels');
  const metricParisClass = document.getElementById('metric-paris-class');
  const metricRiskTier = document.getElementById('metric-risk-tier');
  const metricRiskLabel = document.getElementById('metric-risk-label');
  const metricLatency = document.getElementById('metric-latency');
  const metricHardware = document.getElementById('metric-hardware');
  const recommendationText = document.getElementById('recommendation-text');

  // Modals & Action Buttons
  const btnModelInfo = document.getElementById('btn-model-info');
  const modalModel = document.getElementById('modal-model');
  const btnCloseModelModal = document.getElementById('btn-close-model-modal');

  const btnPrintReport = document.getElementById('btn-print-report');
  const modalReport = document.getElementById('modal-report');
  const btnCloseReportModal = document.getElementById('btn-close-report-modal');

  const btnStreamSim = document.getElementById('btn-stream-sim');
  const streamLiveBanner = document.getElementById('stream-live-banner');
  const btnStopStream = document.getElementById('btn-stop-stream');

  const btnZoomIn = document.getElementById('btn-zoom-in');
  const btnZoomOut = document.getElementById('btn-zoom-out');
  const btnZoomReset = document.getElementById('btn-zoom-reset');
  const btnFullscreen = document.getElementById('btn-fullscreen');

  const btnDownloadOverlay = document.getElementById('btn-download-overlay');
  const btnDownloadMask = document.getElementById('btn-download-mask');
  const btnCopyJson = document.getElementById('btn-copy-json');

  const toast = document.getElementById('toast');
  const toastMsg = document.getElementById('toast-msg');

  // =========================================================================
  // 1. Toast Notification Utility
  // =========================================================================
  function showToast(message, duration = 3000) {
    toastMsg.textContent = message;
    toast.classList.remove('hidden');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => {
      toast.classList.add('hidden');
    }, duration);
  }

  // =========================================================================
  // 2. Interactive Split Slider Logic
  // =========================================================================
  let isDraggingSplit = false;

  function updateSplitPosition(clientX) {
    if (!splitWrapper) return;
    const rect = splitWrapper.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const percent = (x / rect.width) * 100;

    splitDivider.style.left = `${percent}%`;
    splitTopWrapper.style.width = `${percent}%`;
    splitWrapper.style.setProperty('--viewport-width', `${rect.width}px`);
  }

  splitWrapper.addEventListener('mousedown', (e) => {
    isDraggingSplit = true;
    updateSplitPosition(e.clientX);
  });

  window.addEventListener('mousemove', (e) => {
    if (isDraggingSplit) {
      updateSplitPosition(e.clientX);
    }
  });

  window.addEventListener('mouseup', () => {
    isDraggingSplit = false;
  });

  // Touch support for split slider
  splitWrapper.addEventListener('touchstart', (e) => {
    isDraggingSplit = true;
    if (e.touches.length > 0) updateSplitPosition(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (isDraggingSplit && e.touches.length > 0) {
      updateSplitPosition(e.touches[0].clientX);
    }
  }, { passive: true });

  window.addEventListener('touchend', () => {
    isDraggingSplit = false;
  });

  // =========================================================================
  // 3. View Mode Switching Logic
  // =========================================================================
  function setViewMode(mode) {
    state.currentMode = mode;

    modeTabs.forEach(tab => {
      if (tab.dataset.mode === mode) {
        tab.classList.add('active');
      } else {
        tab.classList.remove('active');
      }
    });

    // Toggle viewport visibility
    if (mode === 'split') {
      viewSplit.classList.remove('hidden');
      viewSingle.classList.add('hidden');
      viewQuad.classList.add('hidden');
    } else if (mode === 'quad') {
      viewSplit.classList.add('hidden');
      viewSingle.classList.add('hidden');
      viewQuad.classList.remove('hidden');
      updateQuadImages();
    } else {
      // Single Layer mode (overlay, contours, heatmap, mask)
      viewSplit.classList.add('hidden');
      viewSingle.classList.remove('hidden');
      viewQuad.classList.add('hidden');
      updateSingleViewLayer(mode);
    }
  }

  function updateSingleViewLayer(mode) {
    if (!state.lastAnalysis) return;
    const imgs = state.lastAnalysis.images;

    if (mode === 'overlay') {
      singleDisplayImg.src = imgs.overlay;
      singleLayerTag.textContent = 'Alpha-Blend Overlay';
    } else if (mode === 'contours') {
      singleDisplayImg.src = imgs.contours;
      singleLayerTag.textContent = 'Boundary Contours & Centroid';
    } else if (mode === 'heatmap') {
      singleDisplayImg.src = imgs.heatmap;
      singleLayerTag.textContent = 'Probability Heatmap (Jet)';
    } else if (mode === 'mask') {
      singleDisplayImg.src = imgs.mask;
      singleLayerTag.textContent = 'Binary Segmentation Mask';
    }
  }

  function updateQuadImages() {
    if (!state.lastAnalysis) return;
    const imgs = state.lastAnalysis.images;
    quadRaw.src = imgs.original;
    quadOverlay.src = imgs.overlay;
    quadHeatmap.src = imgs.heatmap;
    quadMask.src = imgs.mask;
  }

  modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      setViewMode(tab.dataset.mode);
    });
  });

  // =========================================================================
  // 4. API Inference Runner
  // =========================================================================
  async function runInference(formData) {
    if (state.isAnalyzing) return;
    state.isAnalyzing = true;
    stageLoader.classList.remove('hidden');
    scanline.classList.remove('hidden');

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.status === 'success') {
        state.lastAnalysis = data;
        renderAnalysis(data);
        showToast(`Segmentation Complete (${data.latency_ms} ms)`);
      } else {
        showToast(`Error: ${data.message || 'Inference failed'}`);
      }
    } catch (err) {
      console.error('Inference error:', err);
      showToast('Connection error during inference');
    } finally {
      state.isAnalyzing = false;
      stageLoader.classList.add('hidden');
      scanline.classList.add('hidden');
    }
  }

  // =========================================================================
  // 5. Render Diagnostics & Telemetry
  // =========================================================================
  function renderAnalysis(data) {
    // 1. Update Images in Stage
    splitRawImg.src = data.images.original;
    splitOverlayImg.src = data.images.overlay;

    if (state.currentMode === 'quad') {
      updateQuadImages();
    } else if (state.currentMode !== 'split') {
      updateSingleViewLayer(state.currentMode);
    }

    // 2. Update Primary Diagnostic Alert Banner
    if (data.detected) {
      diagBanner.className = 'diagnostic-banner card status-pos';
      bannerIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
      detectionBadge.className = 'badge-status-pos';
      detectionBadge.textContent = 'POSITIVE: NEOPLASTIC LESION DETECTED';
    } else {
      diagBanner.className = 'diagnostic-banner card status-neg';
      bannerIcon.innerHTML = '<i class="fa-solid fa-shield-check"></i>';
      detectionBadge.className = 'badge-status-neg';
      detectionBadge.textContent = 'NEGATIVE: CLEAR NORMAL MUCOSA';
    }

    parisBadge.textContent = data.paris_class;
    morphologyText.textContent = data.morphology;
    confVal.textContent = data.confidence.toFixed(1);
    confBarFill.style.width = `${data.confidence}%`;

    // 3. Update 4 Telemetry Metric Cards
    metricAreaPct.textContent = `${data.area_pct}%`;
    metricPixels.textContent = data.lesion_pixels.toLocaleString();
    metricTotalPixels.textContent = data.total_pixels.toLocaleString();

    metricParisClass.textContent = data.paris_class.split(' ')[0] || data.paris_class;
    
    // Risk tier
    if (data.detected) {
      metricRiskTier.textContent = data.risk_level.split(':')[0] || 'Tier 2';
      metricRiskTier.className = 'metric-med-val risk-tier-high';
    } else {
      metricRiskTier.textContent = 'Tier 0';
      metricRiskTier.className = 'metric-med-val risk-tier-low';
    }
    metricRiskLabel.textContent = data.risk_level;

    // Latency & Device
    metricLatency.textContent = data.latency_ms;
    if (data.model_info && data.model_info.device) {
      metricHardware.textContent = `${data.model_info.device} • VGG-16 U-Net`;
    }

    // Clinical Recommendation Box
    recommendationText.textContent = data.recommendation;

    // 4. Update Report Modal Data
    updateReportData(data);
  }

  function updateReportData(data) {
    document.getElementById('rep-img-raw').src = data.images.original;
    document.getElementById('rep-img-contour').src = data.images.contours;
    document.getElementById('rep-img-mask').src = data.images.mask;

    document.getElementById('rep-status-val').textContent = data.detected ? 'POSITIVE: LESION DETECTED' : 'NEGATIVE: BENIGN';
    document.getElementById('rep-conf-val').textContent = `${data.confidence}%`;
    document.getElementById('rep-paris-val').textContent = data.paris_class;
    document.getElementById('rep-area-val').textContent = `${data.area_pct}% of FOV`;
    document.getElementById('rep-pixels-val').textContent = data.lesion_pixels.toLocaleString();
    document.getElementById('rep-risk-val').textContent = data.risk_level;
    document.getElementById('rep-rec-text').textContent = data.recommendation;
    document.getElementById('rep-timestamp').textContent = new Date().toLocaleString();
  }

  // =========================================================================
  // 6. Preset Sample Card Selection
  // =========================================================================
  function selectPresetSample(sampleId) {
    state.currentSample = sampleId;

    sampleCards.forEach(c => {
      if (c.dataset.sample === sampleId) {
        c.classList.add('active');
      } else {
        c.classList.remove('active');
      }
    });

    const fd = new FormData();
    fd.append('sample_id', sampleId);
    fd.append('color', state.currentColor);
    fd.append('opacity', state.currentOpacity);
    fd.append('threshold', state.currentThreshold);

    runInference(fd);
  }

  sampleCards.forEach(card => {
    card.addEventListener('click', () => {
      selectPresetSample(card.dataset.sample);
    });
  });

  // =========================================================================
  // 7. File Upload & Drag-and-Drop & Clipboard Paste
  // =========================================================================
  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  // Clipboard paste support (Ctrl+V / Cmd+V)
  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (const item of items) {
      if (item.type.indexOf('image') !== -1) {
        const file = item.getAsFile();
        showToast('Image pasted from clipboard!');
        handleFileUpload(file);
        break;
      }
    }
  });

  function handleFileUpload(file) {
    sampleCards.forEach(c => c.classList.remove('active'));
    state.currentSample = null;

    const fd = new FormData();
    fd.append('file', file);
    fd.append('color', state.currentColor);
    fd.append('opacity', state.currentOpacity);
    fd.append('threshold', state.currentThreshold);

    runInference(fd);
  }

  // =========================================================================
  // 8. Visual Adjustments (Color, Opacity, Sensitivity)
  // =========================================================================
  colorSwatches.forEach(swatch => {
    swatch.addEventListener('click', () => {
      colorSwatches.forEach(s => s.classList.remove('active'));
      swatch.classList.add('active');
      state.currentColor = swatch.dataset.color;
      colorHexLabel.textContent = swatch.title + ' ' + state.currentColor;
      reanalyzeCurrent();
    });
  });

  opacitySlider.addEventListener('input', (e) => {
    const val = e.target.value;
    opacityVal.textContent = `${val}%`;
    state.currentOpacity = val / 100.0;
  });

  opacitySlider.addEventListener('change', () => {
    reanalyzeCurrent();
  });

  thresholdSlider.addEventListener('input', (e) => {
    const val = (e.target.value / 100.0).toFixed(2);
    thresholdVal.textContent = val;
    state.currentThreshold = parseFloat(val);
  });

  thresholdSlider.addEventListener('change', () => {
    reanalyzeCurrent();
  });

  btnReanalyze.addEventListener('click', () => {
    reanalyzeCurrent();
  });

  function reanalyzeCurrent() {
    if (state.currentSample) {
      selectPresetSample(state.currentSample);
    } else if (state.lastAnalysis && state.lastAnalysis.images.original) {
      // Re-run with base64 image
      const fd = new FormData();
      fd.append('color', state.currentColor);
      fd.append('opacity', state.currentOpacity);
      fd.append('threshold', state.currentThreshold);
      
      fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: state.lastAnalysis.images.original,
          color: state.currentColor,
          opacity: state.currentOpacity,
          threshold: state.currentThreshold
        })
      })
      .then(res => res.json())
      .then(data => {
        if (data.status === 'success') {
          state.lastAnalysis = data;
          renderAnalysis(data);
        }
      });
    }
  }

  // =========================================================================
  // 9. Zoom, Fullscreen & Controls
  // =========================================================================
  btnZoomIn.addEventListener('click', () => {
    state.zoomLevel = Math.min(state.zoomLevel + 0.25, 3.0);
    applyZoom();
  });

  btnZoomOut.addEventListener('click', () => {
    state.zoomLevel = Math.max(state.zoomLevel - 0.25, 0.5);
    applyZoom();
  });

  btnZoomReset.addEventListener('click', () => {
    state.zoomLevel = 1.0;
    applyZoom();
  });

  function applyZoom() {
    splitRawImg.style.transform = `translate(-50%, -50%) scale(${state.zoomLevel})`;
    splitOverlayImg.style.transform = `translate(-50%, -50%) scale(${state.zoomLevel})`;
    singleDisplayImg.style.transform = `scale(${state.zoomLevel})`;
  }

  btnFullscreen.addEventListener('click', () => {
    const viewport = document.querySelector('.visualizer-viewport-container');
    viewport.classList.toggle('fullscreen-mode');
    const isFull = viewport.classList.contains('fullscreen-mode');
    btnFullscreen.innerHTML = isFull ? '<i class="fa-solid fa-compress"></i>' : '<i class="fa-solid fa-expand"></i>';
  });

  // =========================================================================
  // 10. Live Video Stream Simulator
  // =========================================================================
  const streamSamples = ['sessile', 'pedunculated', 'flat', 'normal'];

  btnStreamSim.addEventListener('click', () => {
    toggleStreamSimulation();
  });

  btnStopStream.addEventListener('click', () => {
    stopStreamSimulation();
  });

  function toggleStreamSimulation() {
    if (state.isStreaming) {
      stopStreamSimulation();
    } else {
      startStreamSimulation();
    }
  }

  function startStreamSimulation() {
    state.isStreaming = true;
    streamLiveBanner.classList.remove('hidden');
    btnStreamSim.classList.add('active');
    btnStreamSim.innerHTML = '<i class="fa-solid fa-stop"></i> <span>Stop Stream</span>';
    showToast('Real-time colonoscopy video stream simulation started');

    state.streamInterval = setInterval(() => {
      state.streamIndex = (state.streamIndex + 1) % streamSamples.length;
      const nextSample = streamSamples[state.streamIndex];
      selectPresetSample(nextSample);
    }, 2400);
  }

  function stopStreamSimulation() {
    state.isStreaming = false;
    clearInterval(state.streamInterval);
    streamLiveBanner.classList.add('hidden');
    btnStreamSim.classList.remove('active');
    btnStreamSim.innerHTML = '<i class="fa-solid fa-film"></i> <span>Live Stream Sim</span>';
    showToast('Video stream stopped');
  }

  // =========================================================================
  // 11. Modals (Model Specs & Clinical Report)
  // =========================================================================
  btnModelInfo.addEventListener('click', () => {
    modalModel.classList.remove('hidden');
  });

  btnCloseModelModal.addEventListener('click', () => {
    modalModel.classList.add('hidden');
  });

  btnPrintReport.addEventListener('click', () => {
    modalReport.classList.remove('hidden');
  });

  btnCloseReportModal.addEventListener('click', () => {
    modalReport.classList.add('hidden');
  });

  // Close modals on clicking backdrop
  [modalModel, modalReport].forEach(modal => {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.add('hidden');
    });
  });

  // =========================================================================
  // 12. Download & Export Tools
  // =========================================================================
  btnDownloadOverlay.addEventListener('click', () => {
    if (!state.lastAnalysis) return;
    downloadURI(state.lastAnalysis.images.overlay, 'endoscopy_overlay.jpg');
    showToast('Composite overlay downloaded');
  });

  btnDownloadMask.addEventListener('click', () => {
    if (!state.lastAnalysis) return;
    downloadURI(state.lastAnalysis.images.mask, 'endoscopy_binary_mask.png');
    showToast('Binary mask downloaded');
  });

  btnCopyJson.addEventListener('click', () => {
    if (!state.lastAnalysis) return;
    const cleanData = { ...state.lastAnalysis };
    delete cleanData.images; // omit huge base64 strings in clipboard
    navigator.clipboard.writeText(JSON.stringify(cleanData, null, 2)).then(() => {
      showToast('Clinical JSON telemetry copied to clipboard!');
    });
  });

  function downloadURI(uri, name) {
    const link = document.createElement('a');
    link.download = name;
    link.href = uri;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  // =========================================================================
  // 13. Initial Boot
  // =========================================================================
  selectPresetSample('sessile');
});

