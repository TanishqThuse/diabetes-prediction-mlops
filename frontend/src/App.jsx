import React, { useState, useEffect, useCallback } from 'react';
// FIX: Using version-specific import paths for Vite compatibility
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, collection, addDoc, query, orderBy, limit, onSnapshot, serverTimestamp } from 'firebase/firestore';

// Global variables provided by the environment (MANDATORY USE)
const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : null;
const initialAuthToken = typeof __initial_auth_token !== 'undefined' ? __initial_auth_token : null;

// Icons (using inline SVG)
const HeartIcon = ({ className = "w-6 h-6" }) =>v (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
    </svg>
);

const UserIcon = ({ className = "w-4 h-4" }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
    </svg>
);

const LogIcon = ({ className = "w-4 h-4" }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
    </svg>
);

const INITIAL_FORM = {
    Pregnancies: 2,
    Glucose: 130,
    BloodPressure: 70,
    BMI: 28.5,
    Age: 45
};

// Map for display names and validation ranges
const FIELDS_MAP = {
    Pregnancies: { label: "Pregnancies (Count)", min: 0, max: 17, step: 1 },
    Glucose: { label: "Glucose (mg/dL)", min: 44, max: 200, step: 1 },
    BloodPressure: { label: "Blood Pressure (mmHg)", min: 24, max: 122, step: 1 },
    BMI: { label: "BMI (kg/m²)", min: 10.0, max: 67.1, step: 0.1 },
    Age: { label: "Age (Years)", min: 21, max: 81, step: 1 },
};

export default function App() {
    const [form, setForm] = useState(INITIAL_FORM);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [userId, setUserId] = useState(null);
    const [firebaseReady, setFirebaseReady] = useState(false);
    const [history, setHistory] = useState([]);

    // --- Firebase Initialization and Auth ---
    useEffect(() => {
        if (!firebaseConfig) return;

        try {
            const app = initializeApp(firebaseConfig);
            const auth = getAuth(app);
            const dbInstance = getFirestore(app);

            const authenticate = async () => {
                try {
                    if (initialAuthToken) {
                        await signInWithCustomToken(auth, initialAuthToken);
                    } else {
                        await signInAnonymously(auth);
                    }
                } catch (e) {
                    console.error("Firebase Auth Error:", e);
                    setError("Failed to authenticate with Firebase.");
                }
            };

            authenticate();

            // Set up Auth State Listener
            const unsubscribe = onAuthStateChanged(auth, (user) => {
                if (user) {
                    setUserId(user.uid);
                    setFirebaseReady(true);
                } else {
                    // This handles initial anonymous sign-in or sign-inWithCustomToken completion
                    if (!userId) {
                         // Fallback in case onAuthStateChanged fires before signIn call finishes, or fails
                         // Re-running authenticate ensures we have a user/uid
                         authenticate();
                    }
                }
            });

            // Store instances for later use (though we rely on hooks/functions)
            window.db = dbInstance;
            window.auth = auth;
            
            return () => unsubscribe();

        } catch (e) {
            console.error("Firebase Initialization Error:", e);
            setError("Firebase failed to initialize. Check config.");
        }
    }, []);

    // --- Firestore History Listener ---
    useEffect(() => {
        if (!firebaseReady) return;

        const db = window.db;
        
        // Public collection path: /artifacts/{appId}/public/data/diabetes_logs
        const historyCollectionRef = collection(db, 'artifacts', appId, 'public', 'data', 'diabetes_logs');
        
        // Query to get the 10 most recent logs
        const q = query(historyCollectionRef, orderBy('timestamp', 'desc'), limit(10));
        
        const unsubscribe = onSnapshot(q, (snapshot) => {
            const historyList = snapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data(),
                timestamp: doc.data().timestamp ? new Date(doc.data().timestamp.seconds * 1000).toLocaleString() : 'N/A'
            }));
            setHistory(historyList);
        }, (err) => {
            console.error("Firestore Snapshot Error:", err);
            setError("Failed to fetch prediction history.");
        });

        return () => unsubscribe();
    }, [firebaseReady]);


    const handleChange = (e) => {
        let { name, value, type } = e.target;
        
        // Convert to appropriate number type based on field map step
        if (type === 'number') {
            value = FIELDS_MAP[name].step === 1 ? parseInt(value) : parseFloat(value);
        }

        setForm(prev => ({ ...prev, [name]: value }));
    };

    const handlePredict = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setResult(null);

        // Simple validation check for zero values in Glucose/BP
        if (form.Glucose <= 0 || form.BloodPressure <= 0) {
            setError("Glucose and Blood Pressure values must be greater than zero for a valid prediction.");
            setLoading(false);
            return;
        }

        const API_URL = "http://127.0.0.1:8000/predict";
        
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form)
            });

            if (!response.ok) {
                // Throw an error if the HTTP status is 4xx or 5xx
                const errorText = await response.text();
                throw new Error(`API returned status ${response.status}: ${errorText}`);
            }

            const data = await response.json();
            
            // Expected structure: { prediction: "Diabetic" | "Non-Diabetic", probability: 0.XX }
            setResult({
                prediction: data.prediction,
                probability: data.probability || 'N/A' // Handle missing probability gracefully
            });

            // Log the prediction to Firestore (MLOps Feature)
            if (firebaseReady && window.db) {
                const db = window.db;
                const logData = {
                    ...form,
                    prediction_result: data.prediction,
                    probability: data.probability,
                    userId: userId,
                    timestamp: serverTimestamp()
                };
                
                await addDoc(collection(db, 'artifacts', appId, 'public', 'data', 'diabetes_logs'), logData);
            }

        } catch (err) {
            console.error("Prediction or Logging Error:", err);
            setError("Prediction failed. Ensure the FastAPI backend is running on http://127.0.0.1:8000. Error: " + err.message);
        } finally {
            setLoading(false);
        }
    };

    // Helper component for styled input
    const InputField = useCallback(({ name, value, onChange }) => {
        const field = FIELDS_MAP[name];
        return (
            <div className="flex flex-col space-y-1">
                <label htmlFor={name} className="text-sm font-medium text-gray-600 dark:text-gray-300 flex justify-between items-center">
                    {field.label}
                    <span className="text-xs text-indigo-500 font-bold">{value}</span>
                </label>
                <input
                    id={name}
                    type="number"
                    name={name}
                    value={value}
                    onChange={onChange}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    className="p-3 border border-gray-300 rounded-xl shadow-inner bg-white dark:bg-gray-700 dark:border-gray-600 focus:ring-2 focus:ring-indigo-500 transition duration-150"
                    required
                />
            </div>
        );
    }, []);

    // Helper component for Prediction Card
    const PredictionCard = ({ prediction, probability }) => {
        const isDiabetic = prediction === "Diabetic";
        const colorClass = isDiabetic ? "bg-red-500 dark:bg-red-600" : "bg-green-500 dark:bg-green-600";
        const iconColor = isDiabetic ? "text-red-100" : "text-green-100";
        const resultText = isDiabetic ? "High Risk: Diabetes Detected" : "Low Risk: Non-Diabetic";

        return (
            <div className={`mt-8 p-6 ${colorClass} text-white rounded-2xl shadow-xl flex items-center justify-between transition-all duration-500`}>
                <div className="flex items-center space-x-4">
                    <HeartIcon className={`w-10 h-10 ${iconColor}`} />
                    <div>
                        <p className="text-sm opacity-80">Prediction Result</p>
                        <h3 className="text-3xl font-extrabold">{resultText}</h3>
                    </div>
                </div>
                <div className="text-right">
                    <p className="text-sm opacity-80">Confidence</p>
                    <h4 className="text-2xl font-bold">{`${(probability * 100).toFixed(1)}%`}</h4>
                </div>
            </div>
        );
    };


    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 font-sans transition-colors duration-300 p-4 sm:p-8 flex justify-center">
            <div className="w-full max-w-7xl flex flex-col lg:flex-row space-y-8 lg:space-y-0 lg:space-x-8">

                {/* --- LEFT COLUMN: INPUT AND PREDICTION --- */}
                <div className="lg:w-1/2">
                    <div className="bg-white dark:bg-gray-800 rounded-3xl p-6 sm:p-10 shadow-2xl border border-gray-200 dark:border-gray-700">
                        
                        <div className="text-center mb-10">
                            <h1 className="text-4xl font-extrabold text-indigo-600 dark:text-indigo-400">
                                🩺 ML Prediction Interface
                            </h1>
                            <p className="text-gray-500 dark:text-gray-400 mt-2">
                                Predict Diabetes Risk using the MLOps FastAPI backend.
                            </p>
                        </div>

                        <form onSubmit={handlePredict} className="space-y-6">
                            {Object.keys(INITIAL_FORM).map((key) => (
                                <InputField
                                    key={key}
                                    name={key}
                                    value={form[key]}
                                    onChange={handleChange}
                                />
                            ))}
                            
                            {error && (
                                <div className="p-3 bg-red-100 dark:bg-red-900 border border-red-400 text-red-700 dark:text-red-300 rounded-lg text-sm">
                                    {error}
                                </div>
                            )}

                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full py-4 mt-6 bg-indigo-600 text-white rounded-xl text-lg font-bold shadow-lg shadow-indigo-500/50 hover:bg-indigo-700 transition duration-300 disabled:bg-indigo-400 flex items-center justify-center space-x-2"
                            >
                                {loading && (
                                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                )}
                                <span>{loading ? "Analyzing Metrics..." : "Predict Diabetes Risk"}</span>
                            </button>
                        </form>

                        {result && <PredictionCard {...result} />}
                    </div>
                </div>
                
                {/* --- RIGHT COLUMN: MLOPS LOG & HISTORY --- */}
                <div className="lg:w-1/2">
                    <div className="bg-white dark:bg-gray-800 rounded-3xl p-6 sm:p-8 shadow-2xl border border-gray-200 dark:border-gray-700 h-full">
                        <div className="flex items-center space-x-2 mb-6 border-b pb-4 border-gray-200 dark:border-gray-700">
                            <LogIcon className="w-6 h-6 text-indigo-500" />
                            <h2 className="text-2xl font-bold text-gray-800 dark:text-white">
                                Prediction History (MLOps Log)
                            </h2>
                        </div>
                        
                        <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-2">
                            {history.length === 0 ? (
                                <div className="text-center py-10 text-gray-500 dark:text-gray-400">
                                    {firebaseReady ? 'Run your first prediction to start logging data.' : 'Loading MLOps features...'}
                                </div>
                            ) : (
                                history.map((log, index) => (
                                    <div key={log.id} className={`p-4 rounded-xl shadow-sm border ${log.prediction_result === 'Diabetic' ? 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800' : 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800'} transition duration-150`}>
                                        <div className="flex justify-between items-start text-sm">
                                            <span className="font-bold text-gray-800 dark:text-white">
                                                {log.prediction_result}
                                            </span>
                                            <span className="text-xs text-gray-500 dark:text-gray-400">{log.timestamp}</span>
                                        </div>
                                        <p className="text-xs text-gray-600 dark:text-gray-300 mt-1">
                                            <span className="font-semibold">P:</span> {log.Pregnancies}, <span className="font-semibold">G:</span> {log.Glucose}, <span className="font-semibold">BP:</span> {log.BloodPressure}, <span className="font-semibold">BMI:</span> {log.BMI}, <span className="font-semibold">Age:</span> {log.Age}
                                        </p>
                                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                            Confidence: {(log.probability * 100).toFixed(1)}%
                                        </p>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                {/* --- FOOTER/METADATA --- */}
                <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center text-xs text-gray-400 dark:text-gray-600">
                    <p className="flex items-center space-x-1">
                        <span className="font-bold">App ID:</span> <span>{appId}</span>
                    </p>
                    <p className="flex items-center space-x-1">
                        <UserIcon className="w-4 h-4" />
                        <span className="font-bold">User ID:</span> 
                        <span className="truncate max-w-[150px] sm:max-w-none">{userId || 'Authenticating...'}</span>
                    </p>
                </div>
            </div>
        </div>
    );
}

// import React, { useState } from "react";
// import axios from "axios";
// import { motion } from "framer-motion";
// import { Bar } from "react-chartjs-2";
// import { Chart as ChartJS, BarElement, CategoryScale, LinearScale } from "chart.js";
// ChartJS.register(BarElement, CategoryScale, LinearScale);

// export default function App() {
//   const [form, setForm] = useState({
//     Pregnancies: "",
//     Glucose: "",
//     BloodPressure: "",
//     BMI: "",
//     Age: ""
//   });

//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const handleChange = (e) => {
//     setForm({ ...form, [e.target.name]: e.target.value });
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     setLoading(true);
//     try {
//       const res = await axios.post("http://127.0.0.1:8000/predict", form);
//       setResult(res.data);
//     } catch (err) {
//       alert("Prediction failed. Make sure backend is running.");
//     } finally {
//       setLoading(false);
//     }
//   };

//   const data = {
//     labels: ["Non-Diabetic", "Diabetic"],
//     datasets: [
//       {
//         label: "Prediction Confidence",
//         data: result ? [100 - result.probability, result.probability] : [0, 0],
//         backgroundColor: ["#22c55e", "#ef4444"]
//       }
//     ]
//   };

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-blue-100 to-indigo-300 flex items-center justify-center p-6">
//       <motion.div
//         initial={{ opacity: 0, y: 20 }}
//         animate={{ opacity: 1, y: 0 }}
//         transition={{ duration: 0.8 }}
//         className="bg-white rounded-3xl shadow-2xl p-10 w-full max-w-3xl"
//       >
//         <h1 className="text-4xl font-bold text-center mb-8 text-indigo-600">
//           🩺 Diabetes Health Predictor
//         </h1>

//         <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-6">
//           {Object.keys(form).map((key) => (
//             <div key={key} className="flex flex-col">
//               <label className="font-semibold text-gray-700">{key}</label>
//               <input
//                 type="number"
//                 name={key}
//                 value={form[key]}
//                 onChange={handleChange}
//                 className="mt-1 p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400"
//                 required
//               />
//             </div>
//           ))}
//           <div className="col-span-2 flex justify-center">
//             <motion.button
//               whileHover={{ scale: 1.05 }}
//               type="submit"
//               disabled={loading}
//               className="px-6 py-3 bg-indigo-600 text-white rounded-xl text-lg font-semibold shadow-lg"
//             >
//               {loading ? "Predicting..." : "Predict"}
//             </motion.button>
//           </div>
//         </form>

//         {result && (
//           <motion.div
//             initial={{ opacity: 0 }}
//             animate={{ opacity: 1 }}
//             transition={{ duration: 1 }}
//             className="mt-10 text-center"
//           >
//             <h2 className="text-2xl font-bold text-gray-700 mb-4">
//               Result:{" "}
//               <span
//                 className={`${
//                   result.prediction === "Diabetic" ? "text-red-500" : "text-green-500"
//                 }`}
//               >
//                 {result.prediction}
//               </span>
//             </h2>
//             <div className="max-w-md mx-auto">
//               <Bar data={data} />
//             </div>
//           </motion.div>
//         )}
//       </motion.div>
//     </div>
//   );
// }
