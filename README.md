AI-Powered Sentiment Analysis Platform
This full-stack web application enables users to upload text-based files, analyze sentiments across different categories, and interact with a chatbot that generates AI-powered insights. It features advanced sentiment classification, category-specific comment extraction, and a dynamic conversation system — all integrated with secure Azure AD authentication.

🚀 Features
🔐 Azure AD Authentication (MSAL)
Secure access with Microsoft identity integration.

📁 File Upload and Analysis
Upload .txt, .docx, or .xlsx files and perform sentiment analysis by category.

📊 Category-Based Sentiment Classification
Visualizes positive, negative, and neutral comments per selected category.

🤖 Smart Chatbot
Ask questions related to uploaded data. Summaries and answers are generated via Azure OpenAI.

📌 Highlighted Comments Viewer
Displays most positive and most negative comments per selected category.

🧠 AI-Driven Summarization
On-demand summaries generated from category data or conversation history.

🧱 Tech Stack
Frontend:
React
Tailwind CSS
MSAL.js (Azure AD)
Backend:
FastAPI
Azure OpenAI
Pandas
LangChain
Azure Blob Storage
Authentication:
Azure Active Directory
MSAL React for token handling
