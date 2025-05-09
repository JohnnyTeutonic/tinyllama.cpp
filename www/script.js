document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message');
    const sendButton = document.getElementById('send');
    const historyDiv = document.getElementById('history');
    const serverUrl = '/chat'; // Endpoint defined in server.cpp

    let conversationHistory = []; // Store history as {role: 'user'/'assistant', content: '...'} 

    function addMessageToHistory(role, text, isError = false) {
        conversationHistory.push({ role: role, content: text });
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', role === 'user' ? 'user-message' : 'server-message');
        if (isError) {
            messageDiv.classList.add('error-message');
        }
        // Add prefix based on role
        const prefix = role === 'user' ? 'You: ' : 'AI: ';
        messageDiv.textContent = prefix + text; // Display with prefix
        historyDiv.appendChild(messageDiv);
        historyDiv.scrollTop = historyDiv.scrollHeight; // Scroll to the bottom
    }

    async function sendMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText) return; // Don't send empty messages

        addMessageToHistory('user', messageText);
        messageInput.value = ''; // Clear input field

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    user_input: messageText,
                    temperature: 0.1,  // Low temperature for focused responses
                    max_new_tokens: 60,
                    top_k: 40,        // Limit to top 40 tokens
                    top_p: 0.9        // Sample from 90% of probability mass
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            addMessageToHistory('assistant', data.reply);

        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToHistory('assistant', `Error: ${error.message}`, true);
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    const initialAIMessage = historyDiv.querySelector('.server-message');
    if(initialAIMessage) initialAIMessage.textContent = "AI: Hello! How can I help you today?";
    
    messageInput.focus();
}); 