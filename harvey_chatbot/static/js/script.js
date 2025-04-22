// Main JavaScript for Harvey Medical Chatbot
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const patientList = document.getElementById('patientList');
    const patientHeader = document.getElementById('patientHeader');
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const patientSearch = document.getElementById('patientSearch');
    const patientDetailsModal = document.getElementById('patientDetailsModal');
    const closeModal = document.getElementById('closeModal');
    const modalBody = document.getElementById('modalBody');

    // State variables
    let currentPatientId = null;
    let patients = [];

    // Initialize
    loadPatients();

    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    patientSearch.addEventListener('input', filterPatients);
    closeModal.addEventListener('click', () => {
        patientDetailsModal.style.display = 'none';
    });

    // Click outside modal to close
    window.addEventListener('click', (e) => {
        if (e.target === patientDetailsModal) {
            patientDetailsModal.style.display = 'none';
        }
    });

    // Functions
    function loadPatients() {
        fetch('/api/get_patients')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    patients = data.patients;
                    renderPatientList(patients);
                } else {
                    showError('Failed to load patients: ' + data.error);
                }
            })
            .catch(error => {
                showError('Error: ' + error.message);
            });
    }

    function renderPatientList(patientData) {
        patientList.innerHTML = '';
        
        if (patientData.length === 0) {
            patientList.innerHTML = '<div class="no-results">No patients found</div>';
            return;
        }

        patientData.forEach(patient => {
            const card = document.createElement('div');
            card.className = `patient-card ${patient.id === currentPatientId ? 'selected' : ''}`;
            card.dataset.patientId = patient.id;

            // Format the probability indicator
            const probability = patient.covid_probability || 0;
            let probabilityClass = 'low-probability';
            let riskText = 'Low Risk';
            
            if (probability >= 0.7) {
                probabilityClass = 'high-probability';
                riskText = 'High Risk';
            } else if (probability >= 0.4) {
                probabilityClass = 'medium-probability';
                riskText = 'Medium Risk';
            }
            
            // Get the first two symptoms for display
            const mainSymptoms = patient.symptoms.slice(0, 2).join(", ");

            card.innerHTML = `
                <div class="patient-name">
                    ${patient.name}
                    <span class="patient-id">${patient.id}</span>
                </div>
                <div class="patient-details">${patient.age}y, ${patient.gender} • ${mainSymptoms}</div>
                <div class="covid-probability">
                    <span class="risk-label">${riskText}</span>
                    <div class="probability-indicator">
                        <div class="probability-value ${probabilityClass}" style="width: ${probability * 100}%"></div>
                    </div>
                    <span class="probability-percent">${Math.round(probability * 100)}%</span>
                </div>
            `;

            card.addEventListener('click', () => selectPatient(patient.id));
            patientList.appendChild(card);
        });
    }

    function selectPatient(patientId) {
        if (currentPatientId === patientId) return;
        
        // Clear previous chat messages when selecting a new patient
        chatMessages.innerHTML = '';
        
        currentPatientId = patientId;
        
        // Update UI to show selection
        document.querySelectorAll('.patient-card').forEach(card => {
            card.classList.remove('selected');
            if (card.dataset.patientId === patientId) {
                card.classList.add('selected');
            }
        });

        // Enable chat input
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.placeholder = "Ask about this patient...";
        userInput.focus();

        // Load patient details
        fetch(`/api/get_patient/${patientId}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const patient = data.patient;
                    renderPatientHeader(patient);
                    
                    // Create welcome message with suggestion bullets
                    const welcomeMessage = document.createElement('div');
                    welcomeMessage.className = 'message system';
                    
                    // Customize questions based on risk level
                    let customQuestions = [];
                    if (patient.covid_probability >= 0.7) {
                        customQuestions = [
                            "What lab results confirm the high COVID risk?",
                            "What treatment would you recommend for this high-risk case?"
                        ];
                    } else if (patient.covid_probability >= 0.4) {
                        customQuestions = [
                            "What symptoms suggest possible COVID infection?",
                            "What additional tests would you recommend?"
                        ];
                    } else {
                        customQuestions = [
                            "Why is this patient considered low risk for COVID?",
                            "What alternative diagnoses should be considered?"
                        ];
                    }
                    
                    welcomeMessage.innerHTML = `
                        <div class="message-content">
                            <p>I've loaded the data for ${patient.name}. This patient has a ${Math.round(patient.covid_probability * 100)}% probability of COVID-19 based on their symptoms and lab results.</p>
                            
                            <div class="quick-suggestions">
                                <p class="suggestion-title">Suggested questions:</p>
                                <div class="suggestion-bullets">
                                    <div class="suggestion-bullet" data-question="${customQuestions[0]}">${customQuestions[0]}</div>
                                    <div class="suggestion-bullet" data-question="${customQuestions[1]}">${customQuestions[1]}</div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    chatMessages.appendChild(welcomeMessage);
                    scrollToBottom();
                    
                    // Add event listeners to the suggestion bullets
                    welcomeMessage.querySelectorAll('.suggestion-bullet').forEach(bullet => {
                        bullet.addEventListener('click', () => {
                            const question = bullet.dataset.question;
                            userInput.value = question;
                            sendMessage();
                        });
                    });
                } else {
                    showError('Failed to load patient details: ' + data.error);
                }
            })
            .catch(error => {
                showError('Error: ' + error.message);
            });
    }

    function renderPatientHeader(patient) {
        const probability = patient.covid_probability || 0;
        let riskLevel = 'low-risk';
        let riskText = 'Low Risk';
        
        if (probability >= 0.7) {
            riskLevel = 'high-risk';
            riskText = 'High Risk';
        } else if (probability >= 0.4) {
            riskLevel = 'medium-risk';
            riskText = 'Medium Risk';
        }

        patientHeader.innerHTML = `
            <div class="patient-info">
                <h2>${patient.name}</h2>
                <p>${patient.age} years, ${patient.gender} • ID: ${patient.id}</p>
                <div class="patient-quick-stats">
                    <div class="stat-item">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M8 16H4V4H8V16ZM14 16H10V8H14V16ZM20 16H16V10H20V16ZM2 20H22V18H2V20Z" fill="currentColor"/>
                        </svg>
                        ${patient.symptoms.length} symptoms
                    </div>
                    <div class="stat-item">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM9 17H7V10H9V17ZM13 17H11V7H13V17ZM17 17H15V13H17V17Z" fill="currentColor"/>
                        </svg>
                        ${Object.keys(patient.lab_results || {}).length} lab results
                    </div>
                </div>
            </div>
            <div class="header-actions">
                <span class="probability-badge ${riskLevel}">
                    COVID-19: ${Math.round(probability * 100)}% ${riskText}
                </span>
                <button id="viewDetailsButton">View Details</button>
            </div>
        `;

        // Add event listener for view details button
        document.getElementById('viewDetailsButton').addEventListener('click', () => {
            showPatientDetails(patient);
        });
    }

    function showPatientDetails(patient) {
        // Format lab results with indicators for abnormal values
        const labResultsHtml = Object.entries(patient.lab_results || {}).map(([name, value]) => {
            let isAbnormal = false;
            let normalRange = '';
            
            // Check for abnormal values based on common ranges
            switch(name) {
                case "lymphocyte_count":
                    normalRange = "1.0-4.8 K/uL";
                    isAbnormal = value < 1.0 || value > 4.8;
                    break;
                case "c_reactive_protein":
                    normalRange = "<10 mg/L";
                    isAbnormal = value > 10;
                    break;
                case "d_dimer":
                    normalRange = "<0.5 ug/mL";
                    isAbnormal = value > 0.5;
                    break;
                case "ferritin":
                    normalRange = "12-300 ng/mL";
                    isAbnormal = value > 300 || value < 12;
                    break;
                default:
                    normalRange = "N/A";
            }
            
            return `
                <div class="lab-item">
                    <div class="lab-name">${formatLabName(name)}</div>
                    <div class="lab-value ${isAbnormal ? 'abnormal' : ''}">
                        ${value} ${getLabUnit(name)}
                        <small>(${normalRange})</small>
                    </div>
                </div>
            `;
        }).join('');

        // Format the modal content
        modalBody.innerHTML = `
            <div class="patient-section">
                <h4>Patient Information</h4>
                <p><strong>Name:</strong> ${patient.name}</p>
                <p><strong>Age:</strong> ${patient.age} years</p>
                <p><strong>Gender:</strong> ${patient.gender}</p>
                <p><strong>COVID-19 Probability:</strong> ${Math.round(patient.covid_probability * 100)}%</p>
            </div>
            
            <div class="patient-section">
                <h4>Symptoms</h4>
                <p>${patient.symptoms.join(', ')}</p>
            </div>
            
            <div class="patient-section">
                <h4>Laboratory Results</h4>
                <div class="lab-results">
                    ${labResultsHtml || 'No laboratory results available'}
                </div>
            </div>
            
            <div class="patient-section">
                <h4>Clinical Notes</h4>
                <div class="clinical-notes">${patient.clinical_notes || 'No clinical notes available'}</div>
            </div>
        `;

        patientDetailsModal.style.display = 'block';
    }

    function formatLabName(name) {
        // Convert snake_case to Title Case with proper medical terminology
        const formatted = name.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
        
        // Handle specific lab names
        const labNames = {
            'C Reactive Protein': 'C-Reactive Protein (CRP)',
            'D Dimer': 'D-Dimer',
            'Wbc Count': 'WBC Count',
            'Rbc Count': 'RBC Count'
        };
        
        return labNames[formatted] || formatted;
    }

    function getLabUnit(name) {
        // Return the appropriate unit for each lab test
        const units = {
            'lymphocyte_count': 'K/uL',
            'c_reactive_protein': 'mg/L',
            'd_dimer': 'ug/mL',
            'ferritin': 'ng/mL',
            'wbc_count': 'K/uL',
            'rbc_count': 'M/uL',
            'hemoglobin': 'g/dL',
            'hematocrit': '%',
            'platelet_count': 'K/uL',
            'sodium': 'mmol/L',
            'potassium': 'mmol/L',
            'chloride': 'mmol/L',
            'bicarbonate': 'mmol/L',
            'bun': 'mg/dL',
            'creatinine': 'mg/dL',
            'glucose': 'mg/dL',
            'calcium': 'mg/dL',
            'magnesium': 'mg/dL',
            'phosphorus': 'mg/dL',
            'alt': 'U/L',
            'ast': 'U/L',
            'alkaline_phosphatase': 'U/L',
            'total_bilirubin': 'mg/dL',
            'albumin': 'g/dL',
            'total_protein': 'g/dL'
        };
        
        return units[name] || '';
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message || !currentPatientId) return;
        
        // Add user message to chat
        addUserMessage(message);
        userInput.value = '';
        
        // Show loading indicator
        const loadingId = addSystemMessage('Processing your request...', true);
        
        // Set a timeout for the server response
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Request timed out')), 10000); // 10 second timeout
        });
        
        // Server request promise
        const fetchPromise = fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                patient_id: currentPatientId
            })
        })
        .then(response => response.json());
        
        // Race between timeout and server response
        Promise.race([fetchPromise, timeoutPromise])
            .then(data => {
                // Remove loading message
                removeMessage(loadingId);
                
                if (data.success) {
                    addSystemMessage(data.response);
                } else {
                    showError('Error: ' + data.error);
                    
                    // Show a helpful backup message
                    setTimeout(() => {
                        addSystemMessage("I'm having trouble processing that request. Let me try a different approach...");
                        
                        // Generate a client-side fallback response based on keywords
                        setTimeout(() => {
                            const fallbackResponse = generateClientFallbackResponse(message, currentPatientId);
                            addSystemMessage(fallbackResponse);
                        }, 1500);
                    }, 1000);
                }
            })
            .catch(error => {
                // Remove loading message
                removeMessage(loadingId);
                showError('Error communicating with server: ' + error.message);
                
                // Show a helpful backup message
                setTimeout(() => {
                    addSystemMessage("I'm having trouble communicating with the server. Let me use what I know to help...");
                    
                    // Generate a client-side fallback response based on keywords
                    setTimeout(() => {
                        const fallbackResponse = generateClientFallbackResponse(message, currentPatientId);
                        addSystemMessage(fallbackResponse);
                    }, 1500);
                }, 1000);
            });
    }

    function addUserMessage(text) {
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user';
        messageDiv.innerHTML = `
            <div class="message-content">
                <p>${text}</p>
                <div class="message-time">${timestamp}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function addSystemMessage(text, isLoading = false) {
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const messageId = 'msg-' + Date.now();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        messageDiv.id = messageId;
        
        if (isLoading) {
            messageDiv.innerHTML = `
                <div class="message-content">
                    <p>
                        <span class="loading-animation">Thinking</span>
                    </p>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-content">
                    <p>${text}</p>
                    <div class="message-time">${timestamp}</div>
                </div>
            `;
        }
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
        
        return messageId;
    }

    function removeMessage(messageId) {
        const messageDiv = document.getElementById(messageId);
        if (messageDiv) {
            messageDiv.remove();
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showError(message) {
        addSystemMessage('Error: ' + message);
    }
    
    // Client-side fallback response generator
    function generateClientFallbackResponse(userMessage, patientId) {
        userMessage = userMessage.toLowerCase();
        
        // Find patient in our local cache
        const patient = patients.find(p => p.id === patientId);
        
        if (!patient) {
            return "I'm having trouble accessing patient information. Please try selecting the patient again.";
        }
        
        // Keywords for symptoms
        if (userMessage.includes('symptom') || userMessage.includes('presenting') || userMessage.includes('complain')) {
            return `${patient.name} is presenting with ${patient.symptoms.join(', ')}.`;
        }
        
        // Keywords for COVID probability
        if (userMessage.includes('covid') && (userMessage.includes('probability') || userMessage.includes('likelihood') || userMessage.includes('chance'))) {
            const percentage = Math.round(patient.covid_probability * 100);
            return `Based on the available information, ${patient.name} has a ${percentage}% likelihood of COVID-19.`;
        }
        
        // Keywords for labs
        if (userMessage.includes('lab') || userMessage.includes('test') || userMessage.includes('result')) {
            if (patient.lab_results) {
                const labNames = Object.keys(patient.lab_results).map(k => k.replace('_', ' ')).slice(0, 3);
                return `${patient.name}'s lab results include ${labNames.join(', ')}. You can ask about specific values.`;
            } else {
                return "I don't see any lab results for this patient in my cached data.";
            }
        }
        
        // Keywords for recommendations
        if (userMessage.includes('recommend') || userMessage.includes('treatment') || userMessage.includes('therapy') || userMessage.includes('plan')) {
            if (patient.covid_probability > 0.7) {
                return "For this high-risk patient, I would recommend COVID-19 testing, isolation precautions, and close monitoring of oxygen levels and respiratory status.";
            } else if (patient.covid_probability > 0.4) {
                return "For this moderate-risk patient, I would recommend COVID-19 testing and symptom monitoring. Home isolation may be appropriate pending test results.";
            } else {
                return "For this lower-risk patient, I would still recommend COVID-19 testing as a precaution, along with monitoring for symptom progression.";
            }
        }
        
        // Default response
        return `I'm analyzing ${patient.name}'s information (${patient.age}y, ${patient.gender}) with symptoms including ${patient.symptoms.join(', ')}. What specific aspect of the case would you like to explore?`;
    }

    function filterPatients() {
        const searchTerm = patientSearch.value.toLowerCase();
        
        if (!searchTerm) {
            renderPatientList(patients);
            return;
        }
        
        const filtered = patients.filter(patient => {
            return (
                patient.name.toLowerCase().includes(searchTerm) ||
                patient.id.toLowerCase().includes(searchTerm) ||
                patient.symptoms.some(s => s.toLowerCase().includes(searchTerm))
            );
        });
        
        renderPatientList(filtered);
    }
});