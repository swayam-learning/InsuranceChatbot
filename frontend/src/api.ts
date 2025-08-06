import axios from "axios";

const API_URL = "https://bajajfinservhackatho-production.up.railway.app/hackrx/run";
const BEARER_TOKEN = "ssbakscstobcb3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757";

export async function sendQuery(pdf: File, query: string): Promise<string> {
  const formData = new FormData();
  formData.append("pdf", pdf);
  formData.append("query", query);

  try {
    const response = await axios.post(API_URL, formData, {
      headers: {
        Authorization: `Bearer ${BEARER_TOKEN}`,
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data.answer;
  } catch (error: any) {
    console.error("Error sending query:", error);
    return "Error: Unable to get response from server.";
  }
}
