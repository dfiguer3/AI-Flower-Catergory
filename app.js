(function () {
  'use strict';

  var MODEL_JSON = 'model.json';
  var METADATA_JSON = 'metadata.json';

  var statusEl = document.getElementById('status');
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

  btnStart.addEventListener('click', startWebcam);
  btnStop.addEventListener('click', stopWebcam);

  Promise.all([
    fetch(METADATA_JSON).then(function (r) {
      if (!r.ok) throw new Error('metadata.json HTTP ' + r.status);
      return r.json();
    }),
    tf.loadLayersModel(MODEL_JSON),
  ])
    .then(function (results) {
      var meta = results[0];
      model = results[1];
      labels = meta.labels || [];
      if (meta.imageSize) imageSize = meta.imageSize;

      setStatus('Model ready. Upload an image or start the webcam.');
      fileInput.disabled = false;
      btnStart.disabled = false;
    })
    .catch(function (err) {
      setStatus(
        'Failed to load model or metadata. Use a local server (not file://). ' + err.message,
        true
      );
    });
})();
