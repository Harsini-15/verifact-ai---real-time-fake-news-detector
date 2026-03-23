import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API Routes
  app.post("/api/predict", async (req, res) => {
    const { text } = req.body;

    if (!text || text.length < 10) {
      return res.status(400).json({ error: "Text is too short for analysis." });
    }

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      // We use Gemini 3.1 Flash for fast, accurate classification and grounding
      const response = await ai.models.generateContent({
        model: "gemini-3.1-flash-preview",
        contents: `Analyze the following news text for authenticity. 
        Classify it as "REAL" or "FAKE". 
        Provide a confidence score (0-1).
        Extract 3-5 key keywords.
        Provide a brief explanation of why it was classified this way.
        
        News Text: "${text}"`,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: "object",
            properties: {
              prediction: { type: "string", enum: ["REAL", "FAKE"] },
              confidence: { type: "number" },
              keywords: { type: "array", items: { type: "string" } },
              explanation: { type: "string" },
              importantWords: { type: "array", items: { type: "string" }, description: "Words that contributed most to the decision" }
            },
            required: ["prediction", "confidence", "keywords", "explanation", "importantWords"]
          }
        }
      });

      const result = JSON.parse(response.text || "{}");
      
      let articles = [];
      
      // If REAL (or even if FAKE, we might want to show what the real story is), 
      // use Google Search grounding to find trusted sources
      if (result.prediction === "REAL" || result.prediction === "FAKE") {
        const searchPrompt = result.prediction === "REAL" 
          ? `Find 3 trusted news articles about: ${result.keywords.join(", ")}`
          : `Find 3 trusted news articles that debunk or provide context for: ${result.keywords.join(", ")}`;

        const groundingResponse = await ai.models.generateContent({
          model: "gemini-3.1-flash-preview",
          contents: searchPrompt,
          config: {
            tools: [{ googleSearch: {} }]
          }
        });

        // Extract grounding chunks
        const chunks = groundingResponse.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
        articles = chunks
          .filter(chunk => chunk.web)
          .map(chunk => ({
            title: chunk.web.title,
            source: new URL(chunk.web.uri).hostname.replace('www.', ''),
            url: chunk.web.uri
          }))
          .slice(0, 3);
      }

      res.json({
        ...result,
        articles
      });
    } catch (error) {
      console.error("Prediction error:", error);
      res.status(500).json({ error: "Failed to analyze news text." });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
