const express = require('express');
const bodyParser = require('body-parser'); // For parsing JSON in the request body
const { spawn } = require('child_process'); // To run the chatbot.py script

const app = express();
const PORT = process.env.PORT || 3001;

app.use(bodyParser.json()); // Parse JSON in the request body

// Define the route to handle the API request
app.post('/process_query', (req, res) => {
  const { query } = req.body; // Assuming the frontend sends the query as JSON

  // Run the chatbot.py script
  const pythonProcess = spawn('python', ['chatbot.py', query]);

  let output = '';

  // Collect data from the child process (chatbot.py)
  pythonProcess.stdout.on('data', (data) => {
    output += data;
  });

  // Handle process completion
  pythonProcess.on('close', (code) => {
    console.log(`chatbot.py process exited with code ${code}`);
    // Process the output as needed
    const processedResult = `Processed Result: ${output}`;
    res.json({ result: processedResult });
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
