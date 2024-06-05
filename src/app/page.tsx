"use client";
import axios from "axios";
import { useState, useEffect, useRef } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

export default function Component() {
  interface Message {
    question: string;
    answer: string | null;
  }

  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const messageEndRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => scrollToBottom(), [messages]);

  const handleSubmit = async (event: any) => {
    try {
      event.preventDefault();
      setQuestion("");
      const message: Message = { question: question, answer: null };
      setMessages((prevMessages) => [...prevMessages, message]);
      setIsGenerating(true);
      const response = await axios.post("https://lena-rag.onrender.com/", {
        message: question,
      });
      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        updatedMessages[updatedMessages.length - 1].answer = response.data;
        return updatedMessages;
      });
      setIsGenerating(false);
    } catch (error) {
      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        updatedMessages[updatedMessages.length - 1].answer =
          "An error has occured.";
        return updatedMessages;
      });
      setIsGenerating(false);
      console.error("Error:", error);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-50 overflow-y-auto">
      <header className="flex items-center px-4 py-3 bg-gray-900 shadow sticky top-0 left-0 right-0 z-50">
        <div className="flex items-center gap-2">
          <div className="rounded-full bg-white text-black flex items-center justify-center w-8 h-8">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-5 w-5"
            >
              <path d="M12 8V4H8" />
              <rect width="16" height="12" x="4" y="8" rx="2" />
              <path d="M2 14h2" />
              <path d="M20 14h2" />
              <path d="M15 13v2" />
              <path d="M9 13v2" />
            </svg>
          </div>
          <div className="text-lg font-medium">RAGBot</div>
        </div>
      </header>
      <div className="flex-1 flex items-center justify-center p-4 py-8 space-y-4">
        {messages.length !== 0 ? (
          <div className="max-w-2xl w-full space-y-4">
            {messages.map((message, index) => (
              <div className="space-y-4" key={index}>
                {message.question && (
                  <div className="flex items-start gap-3 justify-end">
                    <div
                      key={index}
                      className="bg-blue-800 rounded-lg p-3 max-w-[80%]"
                    >
                      <p>{message.question}</p>
                    </div>
                    <div className="rounded-full bg-gray-200 p-2">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                        className="size-6 w-8 h-8 text-gray-950"
                      >
                        <path
                          fillRule="evenodd"
                          d="M7.5 6a4.5 4.5 0 1 1 9 0 4.5 4.5 0 0 1-9 0ZM3.751 20.105a8.25 8.25 0 0 1 16.498 0 .75.75 0 0 1-.437.695A18.683 18.683 0 0 1 12 22.5c-2.786 0-5.433-.608-7.812-1.7a.75.75 0 0 1-.437-.695Z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                  </div>
                )}

                {isGenerating && index === messages.length - 1 ? (
                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-gray-200 p-2">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                        className="size-6 w-8 h-8 text-gray-950"
                      >
                        <path
                          fillRule="evenodd"
                          d="M9 4.5a.75.75 0 0 1 .721.544l.813 2.846a3.75 3.75 0 0 0 2.576 2.576l2.846.813a.75.75 0 0 1 0 1.442l-2.846.813a3.75 3.75 0 0 0-2.576 2.576l-.813 2.846a.75.75 0 0 1-1.442 0l-.813-2.846a3.75 3.75 0 0 0-2.576-2.576l-2.846-.813a.75.75 0 0 1 0-1.442l2.846-.813A3.75 3.75 0 0 0 7.466 7.89l.813-2.846A.75.75 0 0 1 9 4.5ZM18 1.5a.75.75 0 0 1 .728.568l.258 1.036c.236.94.97 1.674 1.91 1.91l1.036.258a.75.75 0 0 1 0 1.456l-1.036.258c-.94.236-1.674.97-1.91 1.91l-.258 1.036a.75.75 0 0 1-1.456 0l-.258-1.036a2.625 2.625 0 0 0-1.91-1.91l-1.036-.258a.75.75 0 0 1 0-1.456l1.036-.258a2.625 2.625 0 0 0 1.91-1.91l.258-1.036A.75.75 0 0 1 18 1.5ZM16.5 15a.75.75 0 0 1 .712.513l.394 1.183c.15.447.5.799.948.948l1.183.395a.75.75 0 0 1 0 1.422l-1.183.395c-.447.15-.799.5-.948.948l-.395 1.183a.75.75 0 0 1-1.422 0l-.395-1.183a1.5 1.5 0 0 0-.948-.948l-1.183-.395a.75.75 0 0 1 0-1.422l1.183-.395c.447-.15.799-.5.948-.948l.395-1.183A.75.75 0 0 1 16.5 15Z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>

                    <div className="bg-gray-800 rounded-lg p-3 max-w-[80%]">
                      <div className="flex items-center justify-center gap-2 animate-pulse">
                        <div className="w-3 h-3 bg-gray-400 rounded-full animate-[bounce_1s_ease-in-out_infinite]" />
                        <div className="w-3 h-3 bg-gray-400 rounded-full animate-[bounce_1s_ease-in-out_0.33s_infinite]" />
                        <div className="w-3 h-3 bg-gray-400 rounded-full animate-[bounce_1s_ease-in-out_0.66s_infinite]" />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-gray-200 p-2">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                        className="size-6 w-8 h-8 text-gray-950"
                      >
                        <path
                          fillRule="evenodd"
                          d="M9 4.5a.75.75 0 0 1 .721.544l.813 2.846a3.75 3.75 0 0 0 2.576 2.576l2.846.813a.75.75 0 0 1 0 1.442l-2.846.813a3.75 3.75 0 0 0-2.576 2.576l-.813 2.846a.75.75 0 0 1-1.442 0l-.813-2.846a3.75 3.75 0 0 0-2.576-2.576l-2.846-.813a.75.75 0 0 1 0-1.442l2.846-.813A3.75 3.75 0 0 0 7.466 7.89l.813-2.846A.75.75 0 0 1 9 4.5ZM18 1.5a.75.75 0 0 1 .728.568l.258 1.036c.236.94.97 1.674 1.91 1.91l1.036.258a.75.75 0 0 1 0 1.456l-1.036.258c-.94.236-1.674.97-1.91 1.91l-.258 1.036a.75.75 0 0 1-1.456 0l-.258-1.036a2.625 2.625 0 0 0-1.91-1.91l-1.036-.258a.75.75 0 0 1 0-1.456l1.036-.258a2.625 2.625 0 0 0 1.91-1.91l.258-1.036A.75.75 0 0 1 18 1.5ZM16.5 15a.75.75 0 0 1 .712.513l.394 1.183c.15.447.5.799.948.948l1.183.395a.75.75 0 0 1 0 1.422l-1.183.395c-.447.15-.799.5-.948.948l-.395 1.183a.75.75 0 0 1-1.422 0l-.395-1.183a1.5 1.5 0 0 0-.948-.948l-1.183-.395a.75.75 0 0 1 0-1.422l1.183-.395c.447-.15.799-.5.948-.948l.395-1.183A.75.75 0 0 1 16.5 15Z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>

                    <div className="bg-gray-800 rounded-lg p-3 max-w-[80%]">
                      {message.answer}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className=" text-white flex items-center justify-center w-20 h-20 opacity-35">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-20 w-20"
            >
              <path d="M12 8V4H8" />
              <rect width="16" height="12" x="4" y="8" rx="2" />
              <path d="M2 14h2" />
              <path d="M20 14h2" />
              <path d="M15 13v2" />
              <path d="M9 13v2" />
            </svg>
          </div>
        )}
      </div>

      <div ref={messageEndRef}></div>
      <form onSubmit={handleSubmit} className="bg-gray-900 p-4 sticky bottom-0 left-0 right-0 z-50">
        <div className="relative max-w-2xl mx-auto">
          <Textarea
            placeholder="Type your message..."
            className="w-full min-h-[50px] rounded-3xl bg-gray-800 text-gray-50 p-3 pr-12 resize-none"
            rows={1}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleSubmit(e);
              }
            }}
          />
          <Button
            type="submit"
            size="icon"
            className="absolute top-2 right-3 rounded-full w-8 h-8 bg-gray-600"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="size-5"
            >
              <path
                fillRule="evenodd"
                d="M11.47 2.47a.75.75 0 0 1 1.06 0l7.5 7.5a.75.75 0 1 1-1.06 1.06l-6.22-6.22V21a.75.75 0 0 1-1.5 0V4.81l-6.22 6.22a.75.75 0 1 1-1.06-1.06l7.5-7.5Z"
                clipRule="evenodd"
              />
            </svg>
          </Button>
        </div>
      </form>
    </div>
  );
}
