const express = require("express");
const path = require("path");
const { spawn } = require("child_process");
const app = express();

app.set("view engine", "pug");
app.use(express.urlencoded({ extended: true }));
app.use(express.static("public"));

app.get("/", (req, res) => {
  // Pass the query parameter to the view
  res.render("index", { feedback: req.query.feedback });
});

app.post("/analyze", (req, res) => {
  const newsContent = req.body.newsText;

  // Spawn Python process
  const python = spawn("python", ["predict.py", newsContent]);

  python.stdout.on("data", (data) => {
    const response = JSON.parse(data.toString());
    res.render("results", {
      prediction: response.result,
      content: newsContent,
    });
  });

  python.stderr.on("data", (err) => console.error(err.toString()));
});

app.post("/feedback", (req, res) => {
  const { content, label, isCorrect } = req.body;

  if (isCorrect === "false") {
    const trueLabel = label === "FAKE" ? "REAL" : "FAKE";
    const python = spawn("python", ["update_model.py", content, trueLabel]);

    python.on("close", (code) => {
      console.log(`Model updated successfully.`);
    });
  }

  // Redirect back to home with a query parameter
  res.redirect("/?feedback=received");
});

app.listen(3000, () => console.log("Server running on http://localhost:3000"));
