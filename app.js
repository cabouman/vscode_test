const fileInput = document.getElementById('fileInput');
const urlInput = document.getElementById('urlInput');
const loadUrlBtn = document.getElementById('loadUrlBtn');
const kernelSize = document.getElementById('kernelSize');
const kernelLabel = document.getElementById('kernelLabel');
const sharpenAmount = document.getElementById('sharpenAmount');
const sharpenLabel = document.getElementById('sharpenLabel');
const sigmaLabel = document.getElementById('sigmaLabel');
const pixelLabel = document.getElementById('pixelLabel');
const resetBtn = document.getElementById('resetBtn');
const downloadBtn = document.getElementById('downloadBtn');
const statusEl = document.getElementById('status');

const sourceCanvas = document.getElementById('sourceCanvas');
const outputCanvas = document.getElementById('outputCanvas');
const sourceCtx = sourceCanvas.getContext('2d');
const outputCtx = outputCanvas.getContext('2d');

let originalImageData = null;
let currentImageData = null;
let isProcessing = false;

const MAX_DIMENSION = 1400;

function setStatus(message) {
  statusEl.textContent = message;
}

function formatKernel(size) {
  kernelLabel.textContent = `${size} x ${size}`;
  const sigma = Math.max(0.15, size / 6);
  sigmaLabel.textContent = sigma.toFixed(2);
  return sigma;
}

function formatSharpen(amount) {
  sharpenLabel.textContent = `${amount.toFixed(1)}x`;
}

function updatePixelLabel() {
  if (!originalImageData) {
    pixelLabel.textContent = '0';
    return;
  }
  pixelLabel.textContent = `${originalImageData.width} x ${originalImageData.height}`;
}

function resizeCanvas(canvas, width, height) {
  canvas.width = width;
  canvas.height = height;
}

function drawEmptyCanvases() {
  resizeCanvas(sourceCanvas, 640, 400);
  resizeCanvas(outputCanvas, 640, 400);
  sourceCtx.fillStyle = '#f1f1f3';
  sourceCtx.fillRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  outputCtx.fillStyle = '#f1f1f3';
  outputCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
}

function applyImage(img) {
  const scale = Math.min(
    1,
    MAX_DIMENSION / Math.max(img.naturalWidth || img.width, img.naturalHeight || img.height)
  );
  const width = Math.round((img.naturalWidth || img.width) * scale);
  const height = Math.round((img.naturalHeight || img.height) * scale);

  resizeCanvas(sourceCanvas, width, height);
  resizeCanvas(outputCanvas, width, height);

  sourceCtx.clearRect(0, 0, width, height);
  sourceCtx.drawImage(img, 0, 0, width, height);

  originalImageData = sourceCtx.getImageData(0, 0, width, height);
  currentImageData = originalImageData;

  outputCtx.putImageData(originalImageData, 0, 0);
  updatePixelLabel();

  downloadBtn.disabled = false;

  if (scale < 1) {
    setStatus(`Loaded and scaled to ${width} x ${height} for real-time preview.`);
  } else {
    setStatus('Image loaded.');
  }

  runFilter();
}

function loadImageFromFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => applyImage(img);
    img.onerror = () => setStatus('Unable to read that image file.');
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

function loadImageFromUrl(url) {
  if (!url) return;
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => applyImage(img);
  img.onerror = () =>
    setStatus('Could not load image. Check the URL or CORS permissions.');
  img.src = url;
}

function buildKernel(size, sigma) {
  const radius = Math.floor(size / 2);
  const kernel = new Float32Array(size);
  const sigma2 = sigma * sigma;
  let sum = 0;

  for (let i = -radius; i <= radius; i += 1) {
    const value = Math.exp(-(i * i) / (2 * sigma2));
    kernel[i + radius] = value;
    sum += value;
  }

  for (let i = 0; i < size; i += 1) {
    kernel[i] /= sum;
  }

  return { kernel, radius };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function gaussianBlur(imageData, size, sigma) {
  const { width, height, data } = imageData;
  const { kernel, radius } = buildKernel(size, sigma);
  const temp = new Float32Array(data.length);
  const output = new Uint8ClampedArray(data.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;

      for (let k = -radius; k <= radius; k += 1) {
        const px = clamp(x + k, 0, width - 1);
        const idx = (y * width + px) * 4;
        const weight = kernel[k + radius];

        r += data[idx] * weight;
        g += data[idx + 1] * weight;
        b += data[idx + 2] * weight;
        a += data[idx + 3] * weight;
      }

      const outIdx = (y * width + x) * 4;
      temp[outIdx] = r;
      temp[outIdx + 1] = g;
      temp[outIdx + 2] = b;
      temp[outIdx + 3] = a;
    }
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;

      for (let k = -radius; k <= radius; k += 1) {
        const py = clamp(y + k, 0, height - 1);
        const idx = (py * width + x) * 4;
        const weight = kernel[k + radius];

        r += temp[idx] * weight;
        g += temp[idx + 1] * weight;
        b += temp[idx + 2] * weight;
        a += temp[idx + 3] * weight;
      }

      const outIdx = (y * width + x) * 4;
      output[outIdx] = r;
      output[outIdx + 1] = g;
      output[outIdx + 2] = b;
      output[outIdx + 3] = a;
    }
  }

  return new ImageData(output, width, height);
}

function sharpenImage(imageData, amount) {
  if (amount <= 0) return imageData;

  const { width, height, data } = imageData;
  const output = new Uint8ClampedArray(data.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (y * width + x) * 4;

      for (let channel = 0; channel < 3; channel += 1) {
        const center = data[idx + channel];
        const left = data[(y * width + clamp(x - 1, 0, width - 1)) * 4 + channel];
        const right = data[(y * width + clamp(x + 1, 0, width - 1)) * 4 + channel];
        const top = data[(clamp(y - 1, 0, height - 1) * width + x) * 4 + channel];
        const bottom = data[(clamp(y + 1, 0, height - 1) * width + x) * 4 + channel];

        const sharpened = center * (1 + amount * 4) - amount * (left + right + top + bottom);
        output[idx + channel] = clamp(Math.round(sharpened), 0, 255);
      }

      output[idx + 3] = data[idx + 3];
    }
  }

  return new ImageData(output, width, height);
}

function runFilter() {
  if (!originalImageData || isProcessing) return;

  const size = Number(kernelSize.value);
  const sigma = formatKernel(size);
  const sharpen = Number(sharpenAmount.value);
  formatSharpen(sharpen);

  if (size <= 1 && sharpen <= 0) {
    outputCtx.putImageData(originalImageData, 0, 0);
    currentImageData = originalImageData;
    return;
  }

  isProcessing = true;
  setStatus('Filtering...');

  requestAnimationFrame(() => {
    try {
      const blurred = size > 1 ? gaussianBlur(originalImageData, size, sigma) : originalImageData;
      const filtered = sharpen > 0 ? sharpenImage(blurred, sharpen) : blurred;

      outputCtx.putImageData(filtered, 0, 0);
      currentImageData = filtered;
      setStatus('Filter applied.');
    } catch (error) {
      console.error(error);
      setStatus('Filtering failed. Try smaller settings.');
    } finally {
      isProcessing = false;
    }
  });
}

fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  loadImageFromFile(file);
});

loadUrlBtn.addEventListener('click', () => {
  loadImageFromUrl(urlInput.value.trim());
});

urlInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    loadImageFromUrl(urlInput.value.trim());
  }
});

kernelSize.addEventListener('input', () => {
  formatKernel(Number(kernelSize.value));
  runFilter();
});

sharpenAmount.addEventListener('input', () => {
  formatSharpen(Number(sharpenAmount.value));
  runFilter();
});

resetBtn.addEventListener('click', () => {
  kernelSize.value = 1;
  sharpenAmount.value = 0;
  formatKernel(1);
  formatSharpen(0);
  runFilter();
});

downloadBtn.addEventListener('click', () => {
  if (!currentImageData) return;
  const link = document.createElement('a');
  link.download = 'gaussian-filtered.png';
  link.href = outputCanvas.toDataURL('image/png');
  link.click();
});

formatKernel(Number(kernelSize.value));
formatSharpen(Number(sharpenAmount.value));
updatePixelLabel();
drawEmptyCanvases();
