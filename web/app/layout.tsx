import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "PAISA — Payment AI for Smart Authentication",
  description: "Real-time fraud detection and intelligent payment routing. Trained on 590,000 IEEE-CIS transactions.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.variable} style={{ fontFamily: "Inter, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}