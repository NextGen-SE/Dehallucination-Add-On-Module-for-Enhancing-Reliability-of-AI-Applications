import "./globals.css";

export const metadata = {
  title: "TruthLayer",
  description: "Dehallucination add-on: claim-level verification of AI-generated text",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
