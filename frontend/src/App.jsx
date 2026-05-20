import { useEffect, useRef, useState } from "react";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "";
const turnstileSiteKey = import.meta.env.VITE_TURNSTILE_SITE_KEY || "";
const disableTurnstile = import.meta.env.VITE_DISABLE_TURNSTILE === "true";

function App() {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);
  const [captchaToken, setCaptchaToken] = useState("");
  const [captchaReady, setCaptchaReady] = useState(false);
  const [captchaError, setCaptchaError] = useState("");
  const turnstileContainerRef = useRef(null);
  const turnstileWidgetIdRef = useRef(null);

  useEffect(() => {
    if (disableTurnstile) {
      setCaptchaReady(true);
      setCaptchaError("");
      return undefined;
    }

    if (!turnstileSiteKey) {
      setCaptchaError("Turnstile is not configured yet. Add VITE_TURNSTILE_SITE_KEY to the frontend environment.");
      return undefined;
    }

    const renderWidget = () => {
      if (!window.turnstile || !turnstileContainerRef.current || turnstileWidgetIdRef.current !== null) {
        return;
      }

      turnstileWidgetIdRef.current = window.turnstile.render(turnstileContainerRef.current, {
        sitekey: turnstileSiteKey,
        callback: (token) => {
          setCaptchaToken(token);
          setCaptchaReady(true);
          setCaptchaError("");
        },
        "expired-callback": () => {
          setCaptchaToken("");
          setCaptchaReady(false);
          setCaptchaError("Verification expired. Please complete it again.");
        },
        "error-callback": () => {
          setCaptchaToken("");
          setCaptchaReady(false);
          setCaptchaError("Verification could not be completed. Please reload and try again.");
        },
      });
    };

    const existingScript = document.querySelector('script[data-turnstile="true"]');
    if (existingScript) {
      existingScript.addEventListener("load", renderWidget, { once: true });
      renderWidget();
      return undefined;
    }

    const script = document.createElement("script");
    script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    script.async = true;
    script.defer = true;
    script.dataset.turnstile = "true";
    script.onload = renderWidget;
    script.onerror = () => {
      setCaptchaError("Cloudflare Turnstile failed to load. Please reload the page.");
    };
    document.head.appendChild(script);

    return undefined;
  }, []);

  const resetTurnstile = () => {
    if (disableTurnstile) {
      return;
    }

    if (window.turnstile && turnstileWidgetIdRef.current !== null) {
      window.turnstile.reset(turnstileWidgetIdRef.current);
    }
    setCaptchaToken("");
    setCaptchaReady(false);
  };

  const sendMessage = async () => {
    if (!message.trim()) return;
    if (!disableTurnstile && !captchaToken) {
      setCaptchaError("Please complete the verification challenge before sending a message.");
      return;
    }

    const userMessage = {
      role: "user",
      text: message
    };

    setChat(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await fetch(`${apiBaseUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: message,
          turnstileToken: captchaToken
        })
      });

      const data = await response.json();

      const botMessage = {
        role: "bot",
        text: data.response
      };

      setChat(prev => [...prev, botMessage]);
      resetTurnstile();

    } catch (error) {
      setChat(prev => [
        ...prev,
        {
          role: "bot",
          text: error.message
        }
      ]);
    }

    setLoading(false);
    setMessage("");
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-white flex justify-center p-6">

      <div className="w-full max-w-3xl">

        <h1 className="text-4xl font-bold mb-6 text-center">
          Wikipedia RAG Chatbot
        </h1>

        <p className="mb-4 text-center text-sm text-zinc-400">
          {disableTurnstile
            ? "Local development mode is enabled with Turnstile disabled."
            : "Public demo mode is enabled with bot checks and daily usage limits."}
        </p>

        <div className="bg-zinc-900 rounded-2xl p-4 h-[70vh] overflow-y-auto shadow-lg">

          {chat.map((msg, index) => (

            <div
              key={index}
              className={`mb-4 flex ${
                msg.role === "user"
                  ? "justify-end"
                  : "justify-start"
              }`}
            >

              <div
                className={`max-w-[80%] px-4 py-3 rounded-2xl ${
                  msg.role === "user"
                    ? "bg-blue-600"
                    : "bg-zinc-800"
                }`}
              >
                {msg.text}
              </div>

            </div>
          ))}

          {loading && (
            <div className="text-zinc-400">
              Thinking...
            </div>
          )}

        </div>

        <div className="flex mt-4 gap-3">

          <input
            type="text"
            className="flex-1 bg-zinc-800 rounded-xl px-4 py-3 outline-none"
            placeholder="Ask a Wikipedia question..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />

          <button
            onClick={sendMessage}
            disabled={loading || (!disableTurnstile && !captchaReady)}
            className="bg-blue-600 px-6 py-3 rounded-xl hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-900"
          >
            Send
          </button>

        </div>

        {!disableTurnstile && (
          <div className="mt-4 rounded-2xl bg-zinc-900 p-4 shadow-lg">
            <div ref={turnstileContainerRef} />
            {captchaError && (
              <p className="mt-3 text-sm text-amber-300">
                {captchaError}
              </p>
            )}
          </div>
        )}

      </div>

    </div>
  );
}

export default App;
