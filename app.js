(function () {
  'use strict';

  var MODEL_JSON = 'model.json';
  var METADATA_JSON = 'metadata.json';

  var statusEl = document.getElementById('status');
  var modelFilesInput = document.getElementById('model-files-input');
  var fileInput = document.getElementById('file-input');
  var previewWrap = document.getElementById('preview-wrap');
  var webcamWrap = document.getElementById('webcam-wrap');
  var btnStart = document.getElementById('btn-webcam-start');
  var btnStop = document.getElementById('btn-webcam-stop');
  var topPredEl = document.getElementById('top-pred');
  var barsEl = document.getElementById('bars');

  var model = null;
  var labels = [];
  var imageSize = 224;

  var video = null;
  var webcamRaf = null;
  var lastWebcamPred = 0;
  var WEBCAM_INTERVAL_MS = 150;

  function setStatus(msg, isError) {
    statusEl.textContent = msg;
    statusEl.className = isError ? 'error' : '';
  }

  function isHttpLike() {
    var p = window.location.protocol;
    return p === 'http:' || p === 'https:';
  }

  function readFileAsText(file) {
    return new Promise(function (resolve, reject) {
      var r = new FileReader();
      r.onload = function () {
        resolve(r.result);
      };
      r.onerror = function () {
        reject(r.error);
      };
      r.readAsText(file);
    });
  }

  function applyLoadedModel(m, meta) {
    if (model) {
      model.dispose();
    }
    model = m;
    labels = meta.labels || [];
    if (meta.imageSize) imageSize = meta.imageSize;

    setStatus('Model ready. Upload an image or start the webcam.');
    fileInput.disabled = false;
    btnStart.disabled = false;
    stopWebcam();
  }

  function loadFromFiles(fileList) {
    if (!fileList || fileList.length === 0) return;

    var modelJson = null;
    var weightsBin = null;
    var metaJson = null;

    for (var i = 0; i < fileList.length; i++) {
      var name = fileList[i].name.toLowerCase();
      if (name === 'model.json') modelJson = fileList[i];
      else if (name === 'weights.bin') weightsBin = fileList[i];
      else if (name === 'metadata.json') metaJson = fileList[i];
    }

    if (!modelJson || !weightsBin || !metaJson) {
      setStatus(
        'Select all three files: model.json, weights.bin, and metadata.json (names must match exactly).',
        true
      );
      return;
    }

    setStatus('Loading model from files…');

    readFileAsText(metaJson)
      .then(function (text) {
        var meta = JSON.parse(text);
        return tf.loadLayersModel(tf.io.browserFiles([modelJson, weightsBin])).then(function (m) {
          return { m: m, meta: meta };
        });
      })
      .then(function (o) {
        applyLoadedModel(o.m, o.meta);
      })
      .catch(function (err) {
        setStatus('Failed to load model from files: ' + err.message, true);
      });
  }

  function loadFromHttp() {
    return Promise.all([
      fetch(METADATA_JSON).then(function (r) {
        if (!r.ok) throw new Error('metadata.json HTTP ' + r.status);
        return r.json();
      }),
      tf.loadLayersModel(MODEL_JSON),
    ]).then(function (results) {
      applyLoadedModel(results[1], results[0]);
    });
  }

  /** Teachable Machine image model preprocessing (matches @teachablemachine/image). */
  function preprocess(imageEl) {
    return tf.tidy(function () {
      return tf.browser
        .fromPixels(imageEl)
        .resizeNearestNeighbor([imageSize, imageSize])
        .expandDims(0)
        .toFloat()
        .div(tf.scalar(127.5))
        .sub(tf.scalar(1));
    });
  }

  function renderProbs(probs) {
    var sorted = probs
      .map(function (p, i) {
        return { label: labels[i] || 'Class ' + i, p: p };
      })
      .sort(function (a, b) {
        return b.p - a.p;
      });

    topPredEl.innerHTML =
      'Top: <span>' +
      sorted[0].label +
      '</span> — ' +
      (sorted[0].p * 100).toFixed(1) +
      '%';

    barsEl.innerHTML = '';
    sorted.forEach(function (row) {
      var pct = (row.p * 100).toFixed(1) + '%';
      var div = document.createElement('div');
      div.className = 'bar-row';
      div.innerHTML =
        '<span>' +
        row.label +
        '</span><span class="track"><span class="fill" style="width:' +
        row.p * 100 +
        '%"></span></span><span class="pct">' +
        pct +
        '</span>';
      barsEl.appendChild(div);
    });
  }

  function predictFromPixels(imageEl) {
    var input = preprocess(imageEl);
    var pred = model.predict(input);
    var data = pred.dataSync();
    input.dispose();
    pred.dispose();
    var probs = Array.prototype.slice.call(data);
    renderProbs(probs);
  }

  function stopWebcam() {
    if (webcamRaf != null) {
      cancelAnimationFrame(webcamRaf);
      webcamRaf = null;
    }
    if (video && video.srcObject) {
      video.srcObject.getTracks().forEach(function (t) {
        t.stop();
      });
      video.srcObject = null;
    }
    if (video && video.parentNode) {
      video.parentNode.removeChild(video);
    }
    video = null;
    webcamWrap.innerHTML = '';
    btnStart.disabled = !model;
    btnStop.disabled = true;
  }

  function webcamLoop() {
    webcamRaf = requestAnimationFrame(webcamLoop);
    if (!video || video.readyState < 2) return;
    var now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (now - lastWebcamPred < WEBCAM_INTERVAL_MS) return;
    lastWebcamPred = now;
    try {
      predictFromPixels(video);
    } catch (e) {
      setStatus('Webcam prediction error: ' + e.message, true);
      stopWebcam();
    }
  }

  function startWebcam() {
    stopWebcam();
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then(function (stream) {
        video = document.createElement('video');
        video.setAttribute('playsinline', '');
        video.setAttribute('autoplay', '');
        video.muted = true;
        video.srcObject = stream;
        webcamWrap.appendChild(video);
        return video.play();
      })
      .then(function () {
        btnStart.disabled = true;
        btnStop.disabled = false;
        lastWebcamPred = 0;
        webcamLoop();
      })
      .catch(function (err) {
        setStatus('Could not access webcam: ' + err.message, true);
      });
  }

  fileInput.addEventListener('change', function () {
    var file = fileInput.files && fileInput.files[0];
    if (!file || !model) return;
    previewWrap.innerHTML = '';
    var img = new Image();
    img.alt = 'Preview';
    var url = URL.createObjectURL(file);
    img.onload = function () {
      URL.revokeObjectURL(url);
      previewWrap.appendChild(img);
      try {
        predictFromPixels(img);
      } catch (e) {
        setStatus('Prediction error: ' + e.message, true);
      }
    };
    img.onerror = function () {
      URL.revokeObjectURL(url);
      setStatus('Could not load image.', true);
    };
    img.src = url;
  });

  modelFilesInput.addEventListener('change', function () {
    loadFromFiles(modelFilesInput.files);
  });

  btnStart.addEventListener('click', startWebcam);
  btnStop.addEventListener('click', stopWebcam);

  if (isHttpLike()) {
    setStatus('Loading model…');
    loadFromHttp().catch(function (err) {
      setStatus(
        'Could not load from server: ' +
          err.message +
          '. Use “Load model files” below, or check that model files are next to index.html.',
        true
      );
    });
  } else {
    setStatus(
      'Opened as ' +
        window.location.protocol +
        '// — select model.json, weights.bin, and metadata.json under Load model files.'
    );
  }
})();
