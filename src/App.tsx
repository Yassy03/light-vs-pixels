import React, { useRef, useState, useCallback, useEffect } from 'react';
import { GoogleGenAI } from '@google/genai';

// Initialize the Gemini SDK
const ai = new GoogleGenAI({ apiKey: import.meta.env.VITE_GEMINI_API_KEY });

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const physicsCanvasRef = useRef<HTMLCanvasElement>(null);
  const generativeCanvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [hasCaptured, setHasCaptured] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  const [aiConcept, setAiConcept] = useState<string | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);

  // 1. Initialize Webcam
  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: 'user', width: 640, height: 640 } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => setIsCameraReady(true);
        }
      } catch (err) {
        console.error("Camera error.", err);
      }
    };
    initCamera();
  }, []);

  // 2. The RGB Denoising & Image Reveal Animation (Path B)
  useEffect(() => {
    const canvas = generativeCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const size = 640; 
    canvas.width = size;
    canvas.height = size;

    if (!hasCaptured) {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, size, size);
      return;
    }

    // DIAGRAMMATIC RESOLUTION: We do the math at 64x64 to mimic classic ML dataset images!
    const noiseRes = 64;
    const noiseCanvas = document.createElement('canvas');
    noiseCanvas.width = noiseRes; noiseCanvas.height = noiseRes;
    const nCtx = noiseCanvas.getContext('2d');
    if (!nCtx) return;

    const imgData = nCtx.createImageData(noiseRes, noiseRes);
    const data = imgData.data;

    if (isAnalyzing || (!generatedImage && aiConcept)) {
      // STATE 1: Pure RGB Gaussian Noise
      const drawNoise = () => {
        for(let i = 0; i < data.length; i += 4) {
          data[i] = Math.random() * 255;     
          data[i+1] = Math.random() * 255;   
          data[i+2] = Math.random() * 255;   
          data[i+3] = 255;                   
        }
        nCtx.putImageData(imgData, 0, 0);
        
        ctx.imageSmoothingEnabled = false; 
        ctx.globalAlpha = 1;
        ctx.drawImage(noiseCanvas, 0, 0, size, size);
        
        animationRef.current = requestAnimationFrame(drawNoise);
      };
      drawNoise();
      
    } else if (generatedImage) {
      // STATE 2: True Organic Diffusion Cycle
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      
      const img = new Image();
      img.onload = () => {
        const targetCanvas = document.createElement('canvas');
        targetCanvas.width = noiseRes; targetCanvas.height = noiseRes;
        const tCtx = targetCanvas.getContext('2d');
        if (!tCtx) return;
        tCtx.drawImage(img, 0, 0, noiseRes, noiseRes);
        const targetData = tCtx.getImageData(0, 0, noiseRes, noiseRes).data;

        let step = 0;
        const totalSteps = 90; 

        const denoise = () => {
          step++;
          const progress = step / totalSteps; 
          
          let blockSize = Math.pow(2, Math.floor((1 - progress) * 6)); 
          blockSize = Math.max(1, Math.min(32, blockSize)); // Capped for 64px grid
          
          const noiseScale = Math.pow(1 - progress, 1.2); 
          const signalScale = Math.pow(progress, 0.5); 

          for (let y = 0; y < noiseRes; y++) {
            for (let x = 0; x < noiseRes; x++) {
              const i = (y * noiseRes + x) * 4;
              
              const tY = y - (y % blockSize);
              const tX = x - (x % blockSize);
              const tI = (tY * noiseRes + tX) * 4;
              
              const tR = targetData[tI];
              const tG = targetData[tI+1];
              const tB = targetData[tI+2];
              
              const sR = 128 + (tR - 128) * signalScale;
              const sG = 128 + (tG - 128) * signalScale;
              const sB = 128 + (tB - 128) * signalScale;

              const nR = (Math.random() - 0.5) * 510 * noiseScale;
              const nG = (Math.random() - 0.5) * 510 * noiseScale;
              const nB = (Math.random() - 0.5) * 510 * noiseScale;
              
              data[i]   = Math.max(0, Math.min(255, sR + nR));
              data[i+1] = Math.max(0, Math.min(255, sG + nG));
              data[i+2] = Math.max(0, Math.min(255, sB + nB));
              data[i+3] = 255;
            }
          }
          
          nCtx.putImageData(imgData, 0, 0);
          
          ctx.globalAlpha = 1;
          ctx.imageSmoothingEnabled = false; 
          ctx.drawImage(noiseCanvas, 0, 0, size, size);
          
          if (step < totalSteps) {
            animationRef.current = requestAnimationFrame(denoise);
          } else {
             // FINAL STATE: We deliberately keep the 64x64 pixelated diagram look!
             // We just draw the clean target data without the high-res image overlay.
             for(let i=0; i<data.length; i++) data[i] = targetData[i];
             nCtx.putImageData(imgData, 0, 0);
             ctx.imageSmoothingEnabled = false;
             ctx.drawImage(noiseCanvas, 0, 0, size, size);
          }
        };
        denoise();
      };
      img.src = generatedImage;
    }

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [isAnalyzing, generatedImage, hasCaptured, aiConcept]);

  // 3. The Main Capture Logic
  const captureSubject = useCallback(async () => {
    if (!videoRef.current || !physicsCanvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = physicsCanvasRef.current;
    const vidW = video.videoWidth;
    const vidH = video.videoHeight;
    
    if (vidW === 0 || vidH === 0) return;
    
    const size = Math.min(vidW, vidH);
    const startX = (vidW - size) / 2;
    const startY = (vidH - size) / 2;

    // --- PATH A: POINT CLOUD (PHYSICS) ---
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    if (ctx) {
      ctx.drawImage(video, startX, startY, size, size, 0, 0, size, size);
      const base64Image = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
      
      const imageData = ctx.getImageData(0, 0, size, size);
      const data = imageData.data;
      
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, size, size);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      
      const step = 3; 
      const dotSize = 1.2; 

      for (let y = 0; y < size; y += step) {
        for (let x = 0; x < size; x += step) {
          const i = (y * size + x) * 4;
          const r = data[i]; const g = data[i + 1]; const b = data[i + 2];
          const brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
          const probability = Math.pow(brightness, 2.5);

          if (Math.random() < probability) {
            const jitterX = (Math.random() - 0.5) * 2;
            const jitterY = (Math.random() - 0.5) * 2;
            ctx.fillRect(x + jitterX, y + jitterY, dotSize, dotSize);
          }
        }
      }

      setHasCaptured(true);
      setIsAnalyzing(true);
      setAiConcept(null);
      setGeneratedImage(null);

      // --- PATH B: THE AI CHAIN ---
      try {
        const visionResponse = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: [
            "Describe the main subject in this image in exactly 3 to 6 words. Focus on raw physical description. Do not use punctuation.",
            { inlineData: { data: base64Image, mimeType: "image/jpeg" } }
          ]
        });
        
        const concept = visionResponse.text?.trim() || "Unidentified object";
        setAiConcept(concept); 

        // FIXED PROMPT: No longer asks for documentary photography
        const imageResponse = await ai.models.generateContent({
          model: 'gemini-2.5-flash-image', 
          contents: concept + ", full color photography, vivid, detailed", 
          config: {
            responseModalities: ["IMAGE"],
          }
        });

        const genBase64 = imageResponse.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        
        if (!genBase64) {
          throw new Error("API did not return image data.");
        }
        
        setGeneratedImage(`data:image/jpeg;base64,${genBase64}`);
        setIsAnalyzing(false);
      } catch (err) {
        console.error("AI API Error:", err);
        setAiConcept("Error: System failed to generate image.");
        setIsAnalyzing(false);
      }
    }
  }, []);

  return (
    <div className="min-h-screen bg-[#060606] text-[#eeeeee] font-mono flex flex-col items-center justify-center p-8 selection:bg-white selection:text-black">
      
      <video ref={videoRef} autoPlay playsInline muted className="absolute opacity-0 -z-50 pointer-events-none w-[10px] h-[10px]" />

      <div className="flex flex-col items-center mb-12">
        <h1 className="text-xl tracking-[0.3em] uppercase mb-8 text-white/90">Image Synthesis vs Light Capture</h1>
        <button
          onClick={captureSubject}
          disabled={!isCameraReady || isAnalyzing}
          className="border border-white/40 px-12 py-3 hover:bg-white hover:text-black transition-all duration-300 disabled:opacity-20 disabled:hover:bg-transparent disabled:hover:text-[#eeeeee] uppercase tracking-[0.3em] text-sm font-bold"
        >
          {isCameraReady ? (isAnalyzing ? "Processing..." : "Capture") : "Initializing..."}
        </button>
      </div>

      <div className="flex flex-row gap-16 w-full max-w-5xl justify-center items-start">

        {/* LEFT PATH: PHYSICS */}
        <div className="flex flex-col items-center w-full max-w-[380px]">
          <h2 className="text-sm tracking-[0.3em] uppercase text-white/70 font-bold mb-4">Light Capture</h2>
          <div className="w-full aspect-square border border-white/20 bg-black relative overflow-hidden shadow-2xl shadow-black/50 mb-4">
            <canvas ref={physicsCanvasRef} className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${hasCaptured ? 'opacity-100' : 'opacity-0'}`} />
            <div className="absolute inset-0 pointer-events-none shadow-[inset_0_0_80px_rgba(0,0,0,0.9)] z-10"></div>
            {!hasCaptured && (
              <div className="absolute inset-0 flex items-center justify-center text-white/20 text-xs tracking-widest bg-black z-20">
                [ BLANK SURFACE ]
              </div>
            )}
          </div>
          
          <div className="h-16 flex items-center text-center">
            <span className="text-[10px] text-white/40 tracking-[0.1em] uppercase">
              {hasCaptured ? "Direct optical imprint of physical photons" : ""}
            </span>
          </div>
        </div>

        {/* RIGHT PATH: AI SYNTHESIS */}
        <div className="flex flex-col items-center w-full max-w-[380px]">
          <h2 className="text-sm tracking-[0.3em] uppercase text-white/70 font-bold mb-4">Pixel Synthesis</h2>
          <div className="w-full aspect-square border border-white/20 bg-black relative overflow-hidden shadow-2xl shadow-black/50 mb-4">
            
            <canvas ref={generativeCanvasRef} className={`absolute inset-0 w-full h-full object-cover ${!hasCaptured ? 'opacity-0' : 'opacity-100'}`} />
            
            <div className="absolute inset-0 pointer-events-none shadow-[inset_0_0_80px_rgba(0,0,0,0.9)] z-10"></div>
            
            {!hasCaptured && (
              <div className="absolute inset-0 flex items-center justify-center text-white/20 text-xs tracking-widest bg-black z-20">
                [ AWAITING DATA ]
              </div>
            )}
          </div>
          
          <div className="h-16 flex flex-col items-center justify-center text-center">
             {!hasCaptured ? null : (!generatedImage && aiConcept) ? (
               <span className="text-[10px] tracking-[0.2em] animate-pulse uppercase text-white/50">Synthesizing image from concept...</span>
             ) : isAnalyzing ? (
               <span className="text-[10px] tracking-[0.2em] animate-pulse uppercase text-white/50">Extracting semantic features...</span>
             ) : (
               <>
                 <span className="text-[9px] text-white/30 tracking-[0.2em] uppercase mb-1">Generated From Prompt</span>
                 <span className="text-xs tracking-[0.1em] uppercase font-bold text-white/90">"{aiConcept}"</span>
               </>
             )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default App;