import { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [pdfUrl, setPdfUrl] = useState("");
  const [docxUrl, setDocxUrl] = useState("");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  // ✅ Backend URL (Render)
  const API_BASE_URL =
    import.meta.env.VITE_API_BASE_URL ||
    "https://handwriting-ocr-h4x8.onrender.com";

  const handleUpload = async (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    setFile(selected);
    setPreview(selected.type.startsWith("image") ? URL.createObjectURL(selected) : null);
    setText("");
    setPdfUrl("");
    setDocxUrl("");
    setError("");
    setLoading(true);
    setProgress(0);

    const formData = new FormData();
    formData.append("file", selected);

    try {
      // ✅ Use deployed backend URL
      const res = await axios.post(`${API_BASE_URL}/ocr/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (p) => {
          const percent = Math.round((p.loaded * 100) / p.total);
          setProgress(percent);
        },
      });

      setText(res.data.extracted_text);
      setPdfUrl(res.data.pdf_url);
      setDocxUrl(res.data.docx_url);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || "❌ OCR failed. Please try again.");
    } finally {
      setLoading(false);
      setProgress(100);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isArabic = /[\u0600-\u06FF]/.test(text);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #edf2f7, #cfd9df)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "40px 10px",
        fontFamily: "'Segoe UI', 'Noto Sans Hausa', 'Amiri', sans-serif",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "750px",
          background: "#fff",
          padding: "35px 25px",
          borderRadius: "20px",
          boxShadow: "0 8px 30px rgba(0,0,0,0.1)",
          textAlign: "center",
        }}
      >
        <h1 style={{ color: "#222", marginBottom: "10px" }}>✍️ Multi-Language Handwriting OCR</h1>
        <p style={{ color: "#666", marginBottom: "25px", fontSize: "15px" }}>
          Upload your <b>image</b> or <b>PDF</b> to extract handwritten text.  
          Supports <b>English, French, Hausa & Arabic</b>.
        </p>

        <label
          style={{
            display: "inline-block",
            padding: "12px 24px",
            borderRadius: "8px",
            background: "#007bff",
            color: "white",
            cursor: "pointer",
            fontWeight: "600",
            transition: "all 0.3s",
          }}
        >
          📤 Choose File
          <input
            type="file"
            accept="image/*,.pdf"
            onChange={handleUpload}
            style={{ display: "none" }}
          />
        </label>

        {file && (
          <p style={{ marginTop: "10px", color: "#444" }}>
            Selected: <b>{file.name}</b>
          </p>
        )}

        {preview && (
          <div style={{ marginTop: "20px" }}>
            <img
              src={preview}
              alt="preview"
              style={{
                width: "100%",
                maxHeight: "350px",
                objectFit: "contain",
                borderRadius: "10px",
                boxShadow: "0 2px 12px rgba(0,0,0,0.1)",
              }}
            />
          </div>
        )}

        {loading && (
          <div style={{ marginTop: "25px" }}>
            <p style={{ color: "#007bff", fontWeight: "500" }}>
              ⏳ Extracting text... please wait
            </p>
            <div
              style={{
                width: "100%",
                height: "10px",
                background: "#eee",
                borderRadius: "5px",
                marginTop: "8px",
              }}
            >
              <div
                style={{
                  width: `${progress}%`,
                  height: "10px",
                  background: "#007bff",
                  borderRadius: "5px",
                  transition: "width 0.3s",
                }}
              />
            </div>
          </div>
        )}

        {error && (
          <p
            style={{
              marginTop: "20px",
              background: "#ffe6e6",
              color: "#b30000",
              padding: "12px",
              borderRadius: "8px",
            }}
          >
            {error}
          </p>
        )}

        {text && (
          <div style={{ marginTop: "30px", textAlign: "left" }}>
            <h3 style={{ color: "#333" }}>📜 Extracted Text:</h3>
            <textarea
              readOnly
              dir={isArabic ? "rtl" : "ltr"}
              value={text}
              style={{
                width: "100%",
                height: "250px",
                marginTop: "10px",
                padding: "12px",
                borderRadius: "10px",
                border: "1px solid #ccc",
                fontFamily: isArabic ? "'Amiri', serif" : "monospace",
                background: "#fafafa",
                color: "#222",
                resize: "vertical",
                lineHeight: "1.5",
              }}
            />

            <div
              style={{
                marginTop: "20px",
                display: "flex",
                justifyContent: "center",
                gap: "15px",
                flexWrap: "wrap",
              }}
            >
              <button
                onClick={handleCopy}
                style={{
                  color: "white",
                  background: copied ? "#28a745" : "#17a2b8",
                  padding: "10px 20px",
                  borderRadius: "8px",
                  border: "none",
                  cursor: "pointer",
                  fontWeight: "bold",
                  transition: "background 0.3s",
                }}
              >
                {copied ? "✅ Copied!" : "📋 Copy Text"}
              </button>

              {pdfUrl && (
                <a
                  href={pdfUrl}
                  download
                  style={{
                    color: "white",
                    background: "green",
                    padding: "10px 20px",
                    borderRadius: "8px",
                    textDecoration: "none",
                    fontWeight: "bold",
                  }}
                >
                  📄 Download PDF
                </a>
              )}
              {docxUrl && (
                <a
                  href={docxUrl}
                  download
                  style={{
                    color: "white",
                    background: "#007bff",
                    padding: "10px 20px",
                    borderRadius: "8px",
                    textDecoration: "none",
                    fontWeight: "bold",
                  }}
                >
                  📝 Download Word
                </a>
              )}
            </div>
          </div>
        )}
      </div>

      <p
        style={{
          marginTop: "20px",
          color: "#555",
          fontSize: "14px",
          textAlign: "center",
        }}
      >
        ⚡ Powered by <b>Mbukhary</b> | 📞 <b>08146796232</b>
      </p>
    </div>
  );
}

export default App;
