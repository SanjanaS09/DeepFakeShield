// 

import React, { useState } from 'react';
import {
  Card, CardContent, CardHeader, CardTitle,
  Tabs, TabsContent, TabsList, TabsTrigger,
  Alert, AlertDescription,
  Progress,
  Badge
} from '../components/ui/card';
import { ChevronDown, AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react';

export default function AnalysisResults({ analysisData }) {
  const [expanded, setExpanded] = useState({});
  
  if (!analysisData) {
    return (
      <Alert className="bg-gray-100">
        <AlertDescription>No analysis data available</AlertDescription>
      </Alert>
    );
  }

  const {
    prediction,
    confidence,
    probabilities = {},
    xai = {},
    label
  } = analysisData;

  // ✅ FIX: Ensure valid prediction
  const isPredictionFake = prediction === 'FAKE' || label === 'FAKE';
  const displayPrediction = isPredictionFake ? 'FAKE' : 'REAL';
  const displayConfidence = Math.max(0, Math.min(1, Number(confidence) || 0));

  // ✅ FIX: Get probabilities safely
  const realProb = Number(probabilities.REAL) || (isPredictionFake ? 1 - displayConfidence : displayConfidence);
  const fakeProb = Number(probabilities.FAKE) || (isPredictionFake ? displayConfidence : 1 - displayConfidence);

  return (
    <div className="space-y-6 p-6">
      {/* ✅ MAIN RESULT CARD - FIXED */}
      <Card className={`border-2 ${isPredictionFake ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}`}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              {isPredictionFake ? (
                <>
                  <AlertTriangle className="w-6 h-6 text-red-600" />
                  <span className="text-red-600">🚨 DEEPFAKE DETECTED</span>
                </>
              ) : (
                <>
                  <CheckCircle className="w-6 h-6 text-green-600" />
                  <span className="text-green-600">✅ AUTHENTIC CONTENT</span>
                </>
              )}
            </CardTitle>
            <Badge variant={isPredictionFake ? 'destructive' : 'success'} className="text-lg px-4 py-2">
              {displayPrediction}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Confidence Display */}
          <div>
            <div className="flex justify-between mb-2">
              <span className="font-semibold">Detection Confidence</span>
              <span className="font-bold text-lg">{(displayConfidence * 100).toFixed(1)}%</span>
            </div>
            <Progress 
              value={displayConfidence * 100} 
              className={isPredictionFake ? 'bg-red-200' : 'bg-green-200'}
            />
          </div>

          {/* Confidence Level Badge */}
          <div className="flex gap-2 items-center">
            <span className="text-sm font-semibold">Confidence Level:</span>
            <Badge className={
              displayConfidence > 0.95 ? 'bg-red-600' :
              displayConfidence > 0.80 ? 'bg-red-500' :
              displayConfidence > 0.60 ? 'bg-yellow-500' :
              'bg-gray-500'
            }>
              {xai?.confidence_level || 'Unknown'}
            </Badge>
          </div>

          {/* Probability Breakdown */}
          <div className="grid grid-cols-2 gap-4">
            <div className={`p-4 rounded border-l-4 ${isPredictionFake ? 'bg-green-50 border-green-500' : 'bg-green-100 border-green-600'}`}>
              <div className="text-sm font-semibold text-gray-700">🟢 Real Probability</div>
              <div className="text-3xl font-bold text-green-600 mt-2">
                {!isNaN(realProb) ? (realProb * 100).toFixed(1) : '0.0'}%
              </div>
            </div>
            <div className={`p-4 rounded border-l-4 ${isPredictionFake ? 'bg-red-100 border-red-600' : 'bg-red-50 border-red-500'}`}>
              <div className="text-sm font-semibold text-gray-700">🔴 Fake Probability</div>
              <div className="text-3xl font-bold text-red-600 mt-2">
                {!isNaN(fakeProb) ? (fakeProb * 100).toFixed(1) : '0.0'}%
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ✅ XAI EXPLANATION CARD */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            🔍 Analysis & Reasoning
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="reasoning">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="reasoning">Reasoning</TabsTrigger>
              <TabsTrigger value="indicators">Indicators</TabsTrigger>
              <TabsTrigger value="recommendations">Actions</TabsTrigger>
              <TabsTrigger value="technical">Technical</TabsTrigger>
            </TabsList>

            {/* Reasoning Tab */}
            <TabsContent value="reasoning" className="space-y-4 mt-4">
              <Alert className={isPredictionFake ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}>
                <AlertDescription className="text-base">
                  <div className="font-semibold mb-3">{xai?.explanation}</div>
                  {xai?.reasoning && Array.isArray(xai.reasoning) ? (
                    <ul className="space-y-2 list-disc list-inside">
                      {xai.reasoning.map((reason, idx) => (
                        <li key={idx} className="text-sm">{reason}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm">{xai?.explanation}</p>
                  )}
                </AlertDescription>
              </Alert>
            </TabsContent>

            {/* Indicators Tab */}
            <TabsContent value="indicators" className="space-y-3 mt-4">
              {xai?.key_indicators && typeof xai.key_indicators === 'object' ? (
                Object.entries(xai.key_indicators).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                    <span className="font-semibold text-gray-700 capitalize">
                      {key.replace(/_/g, ' ')}
                    </span>
                    <Badge variant={
                      ['High', 'Natural', 'Minimal', 'Normal', 'Sharp', 'Consistent', 'Stable', 'Safe'].includes(String(value))
                        ? 'default'
                        : 'destructive'
                    }>
                      {String(value)}
                    </Badge>
                  </div>
                ))
              ) : (
                <p className="text-gray-500">No indicators available</p>
              )}
            </TabsContent>

            {/* Recommendations Tab */}
            <TabsContent value="recommendations" className="space-y-2 mt-4">
              {xai?.recommendations && Array.isArray(xai.recommendations) ? (
                <ul className="space-y-2">
                  {xai.recommendations.map((rec, idx) => (
                    <li key={idx} className="flex items-center gap-3 p-2 bg-blue-50 rounded">
                      <span className="text-lg">{rec.split(' ')[0]}</span>
                      <span>{rec.substring(rec.indexOf(' ') + 1)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-500">No recommendations available</p>
              )}
            </TabsContent>

            {/* Technical Tab */}
            <TabsContent value="technical" className="space-y-4 mt-4">
              <Alert>
                <AlertDescription>
                  <div className="font-mono text-sm space-y-2">
                    <p><strong>Model Confidence:</strong> {(displayConfidence * 100).toFixed(4)}%</p>
                    <p><strong>Real Probability:</strong> {(realProb * 100).toFixed(4)}%</p>
                    <p><strong>Fake Probability:</strong> {(fakeProb * 100).toFixed(4)}%</p>
                    <p><strong>Prediction Class:</strong> {displayPrediction}</p>
                    <p><strong>File:</strong> {analysisData.file_name || 'Unknown'}</p>
                  </div>
                </AlertDescription>
              </Alert>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* ✅ FULL JSON DEBUG */}
      <Card className="border-gray-300">
        <CardHeader
          className="cursor-pointer bg-gray-100 hover:bg-gray-200"
          onClick={() => setExpanded(!expanded.debug)}
        >
          <CardTitle className="flex items-center justify-between">
            <span>📋 Full Response Data</span>
            <ChevronDown
              className={`transform transition ${expanded.debug ? 'rotate-180' : ''}`}
              size={20}
            />
          </CardTitle>
        </CardHeader>
        {expanded.debug && (
          <CardContent>
            <pre className="bg-gray-900 text-green-400 p-4 rounded overflow-auto text-xs max-h-96">
              {JSON.stringify(analysisData, null, 2)}
            </pre>
          </CardContent>
        )}
      </Card>
    </div>
  );
}
