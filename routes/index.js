const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');

router.post('/check-news', (req, res) => {
  const newsText = req.body.newsContent;

  // Use a child process to run a python script for prediction
  const pythonProcess = spawn('python', ['./scripts/predict.py', newsText]);

  pythonProcess.stdout.on('data', (data) => {
    const result = data.toString().trim(); // "REAL" or "FAKE"
    res.render('result', { prediction: result, content: newsText });
  });
});

module.exports = router;
