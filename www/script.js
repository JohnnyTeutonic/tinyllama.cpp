document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message');
    const sendButton = document.getElementById('send');
    const historyDiv = document.getElementById('history');
    const serverUrl = '/chat'; // Endpoint defined in server.cpp

    let conversationHistory = []; // Optional: Store history for context

    function addMessageToHistory(text, sender) {
        const messageElement = document.createElement('p');
        messageElement.textContent = text;
        messageElement.classList.add(sender === 'user' ? 'user-message' : 'server-message');
        historyDiv.appendChild(messageElement);
        // Scroll to bottom
        historyDiv.scrollTop = historyDiv.scrollHeight;
    }

    async function sendMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText) return; // Don't send empty messages

        addMessageToHistory(`You: ${messageText}`, 'user');
        messageInput.value = ''; // Clear input
        sendButton.disabled = true; // Disable button while waiting

        // --- Construct Prompt with History (Simple Example) ---
        // let currentPrompt = "";
        // conversationHistory.forEach(entry => {
        //     currentPrompt += `${entry.sender}: ${entry.text}\n`;
        // });
        // currentPrompt += `User: ${messageText}\nAI:`; // Add current message and prompt AI
        // --- For now, just send the last message --- 
        const currentPrompt = messageText; 
        conversationHistory.push({ sender: 'User', text: messageText });

        try {
            const response = await fetch(serverUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: currentPrompt, // Send the potentially constructed prompt
                    // Optional: Send other parameters
                    // temperature: 0.7,
                    // max_new_tokens: 150 
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const reply = data.reply.trim();
            addMessageToHistory(`AI: ${reply}`, 'server');
            conversationHistory.push({ sender: 'AI', text: reply });

        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToHistory(`Error: ${error.message}`, 'server-error'); // Add an error class if needed
        } finally {
            sendButton.disabled = false; // Re-enable button
            messageInput.focus(); // Focus input for next message
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

     // Initial focus
     messageInput.focus();
}); 