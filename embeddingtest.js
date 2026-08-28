const { GoogleGenAI } = require("@google/genai");
require("dotenv").config();

const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
});

async function main() {
    const response = await ai.models.embedContent({
        model: "gemini-embedding-001",
        contents: "Hello world"
    });

    console.log("Embedding size:", response.embeddings[0].values.length);
}

main();