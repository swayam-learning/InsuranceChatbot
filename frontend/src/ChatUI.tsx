import React, { useState } from "react";
import { sendQuery } from "./api";

interface Message {
  sender: "user" | "bot";
  text: string;
}

const ChatUI: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [pdf, setPdf] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPdf(e.target.files[0]);
    }
  };

  const handleSend = async () => {
    if (!query || !pdf) return;
    setMessages([...messages, { sender: "user", text: query }]);
    setLoading(true);

    const answer = await sendQuery(pdf, query);
    setMessages((prev) => [...prev, { sender: "bot", text: answer }]);
    setQuery("");
    setLoading(false);
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 p-4 bg-white shadow-xl rounded-xl">
      <h1 className="text-2xl font-bold mb-4 text-center text-indigo-600">Insurance ChatBot</h1>

      <div className="mb-4">
        <input type="file" accept="application/pdf" onChange={handleFileChange} />
      </div>

      <div className="h-96 overflow-y-auto mb-4 border p-3 rounded bg-gray-50">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`my-2 p-2 rounded ${
              msg.sender === "user" ? "bg-blue-100 text-right" : "bg-green-100 text-left"
            }`}
          >
            {msg.text}
          </div>
        ))}
      </div>

      <div className="flex space-x-2">
        <input
          type="text"
          className="flex-1 border px-3 py-2 rounded"
          placeholder="Ask your query..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700"
          onClick={handleSend}
          disabled={loading}
        >
          {loading ? "Thinking..." : "Send"}
        </button>
      </div>
    </div>
  );
};

export default ChatUI;
