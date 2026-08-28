const express = require('express');
const multer = require('multer');
const PDFParser = require('pdf2json');
const fs = require('fs');
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');
const { QdrantClient } = require("@qdrant/js-client-rest");
require('dotenv').config();

const app = express();
const upload = multer({ dest: 'uploads/' });

// Enable CORS so React client can call this API
app.use(cors());
app.use(express.json());

// Initialize GoogleGenAI SDK
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Choose recommended model for current @google/genai SDK
const MODEL_NAME = 'gemini-3.5-flash';

// Initialize Qdrant Client
const qdrant = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
    checkCompatibility: false
});

// Helper function to extract plain text from PDF using pdf2json
function extractTextFromPDF(filePath) {
    return new Promise((resolve, reject) => {
        const pdfParser = new PDFParser(null, 1);
        pdfParser.on("pdfParser_dataError", errData => reject(errData.parserError));
        pdfParser.on("pdfParser_dataReady", () => {
            const rawText = pdfParser.getRawTextContent();
            resolve(rawText);
        });
        pdfParser.loadPDF(filePath);
    });
}

// Helper function to generate embeddings
async function createEmbedding(text) {
    const response = await ai.models.embedContent({
        model: "gemini-embedding-001",
        contents: text,
    });

    return response.embeddings[0].values;
}

// Helper function to ensure Qdrant collection exists
async function ensureCollection() {
    try {
        const collections = await qdrant.getCollections();

        const exists = collections.collections.some(
            (c) => c.name === "pdf-docs"
        );

        if (!exists) {
            await qdrant.createCollection("pdf-docs", {
                vectors: {
                    size: 3072,
                    distance: "Cosine",
                },
            });

            console.log("Collection created.");
        }
    } catch (err) {
        console.error("Error checking/creating Qdrant collection:", err.message);
    }
}

app.get('/', (req, res) => {
    res.send('ReviseAI Backend API is running!');
});

// 1. Process PDF -> Embeddings in Qdrant + Flashcards & Quiz JSON
app.post('/process-pdf', upload.single('pdf'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No PDF file provided." });
        }

        console.log("Processing uploaded file:", req.file.originalname);
        await ensureCollection();

        // Extract plain text reliably using pdf2json
        const text = await extractTextFromPDF(req.file.path);

        // Clean up temporary uploaded file from disk
        if (fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }

        const chunks = text.split('\n\n').filter((chunk) => chunk.trim() !== "");

        if (chunks.length === 0) {
            return res.status(400).json({ error: "Extracted PDF text is empty." });
        }

        // Store vectors in Qdrant for RAG explanations
        const chunkEmbeddings = [];
        for (const chunk of chunks) {
            const embedding = await createEmbedding(chunk);
            chunkEmbeddings.push({ text: chunk, embedding });
        }

        const points = chunkEmbeddings.map((item, index) => ({
            id: index + 1,
            vector: item.embedding,
            payload: { text: item.text },
        }));

        await qdrant.upsert("pdf-docs", { points });

        // Prompt Gemini to generate structured JSON
        const prompt = `
You are an expert tutor. Analyze the study text below and generate:
1. 5 revision flashcards (term and definition).
2. 5 multiple-choice quiz questions testing key concepts.

Return STRICTLY valid JSON with no markdown formatting, no backticks, and no extra prose:
{
  "flashcards": [
    {
      "id": 1,
      "term": "Concept Name",
      "definition": "Clear concise explanation from the text."
    }
  ],
  "quiz": [
    {
      "question": "Question text here?",
      "option1": "First choice",
      "option2": "Second choice",
      "option3": "Third choice",
      "option4": "Fourth choice",
      "ans": 1
    }
  ]
}

Note: "ans" MUST be an integer (1, 2, 3, or 4) specifying the correct option.

Document Text:
${text.substring(0, 6000)}
`;

        let response;
        try {
            response = await ai.models.generateContent({
                model: MODEL_NAME,
                contents: prompt
            });
        } catch (genErr) {
            if (genErr.status === 429) {
                console.log("Rate limit encountered. Retrying in 12 seconds...");
                await new Promise((resolve) => setTimeout(resolve, 12000));
                response = await ai.models.generateContent({
                    model: MODEL_NAME,
                    contents: prompt
                });
            } else {
                throw genErr;
            }
        }

        let rawText = response.text.trim();
        rawText = rawText.replace(/^```json\s*/i, '').replace(/^```\s*/, '').replace(/\s*```$/, '').trim();
        const studyData = JSON.parse(rawText);

        res.json(studyData);

    } catch (error) {
        console.error("PDF Processing Error:", error);
        res.status(500).json({ error: "Error processing PDF", details: error.message });
    }
});

// 2. RAG Explanation Endpoint for Missed Quiz Questions
app.post('/explain-question', async (req, res) => {
    try {
        const { question, selectedOptionText, correctOptionText } = req.body;

        const queryPrompt = `${question} Correct answer: ${correctOptionText}`;
        const questionEmbedding = await createEmbedding(queryPrompt);

        const queryResult = await qdrant.query("pdf-docs", {
            query: questionEmbedding,
            limit: 1,
            with_payload: true,
        });

        const points = queryResult?.points || queryResult;
        const bestChunk = points[0]?.payload?.text || "No relevant context section found in the PDF.";

        const response = await ai.models.generateContent({
            model: MODEL_NAME,
            contents: `You are an AI study assistant. A student missed a question on their revision quiz.
Question: "${question}"
Student selected: "${selectedOptionText}"
Correct answer: "${correctOptionText}"

Relevant section from their PDF notes:
"${bestChunk}"

In 2-3 concise sentences, explain why "${correctOptionText}" is correct based on the context, and why their choice was incorrect.`
        });

        res.send(response.text);

    } catch (error) {
        console.error("Explanation Error:", error);
        res.status(500).send("Failed to retrieve explanation from PDF context.");
    }
});

app.listen(3000, () => {
    console.log('ReviseAI Backend Server running on http://localhost:3000');
});