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

        // Construct the prompt using the official chat template format
        let promptString = "<|system|>\nYou are a helpful AI.</s>\n"; // Default system prompt
        conversationHistory.forEach(message => {
            if (message.role === 'user') {
                promptString += `<|user|>\n${message.content}</s>\n`;
            } else if (message.role === 'assistant') {
                // Check if the content is not an error message before adding
                if (!message.isError) { // Assuming isError property exists or can be added
                     promptString += `<|assistant|>\n${message.content}</s>\n`;
                }
            }
        });
        promptString += "<|assistant|>\n"; // Add the final assistant marker

        console.log("Sending prompt:", JSON.stringify(promptString)); // Log the exact prompt being sent

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // Send the constructed prompt string directly
                body: JSON.stringify({ message: promptString })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            // Add the AI's response, ensuring it's not marked as an error
            addMessageToHistory('assistant', data.reply);

        } catch (error) {
            console.error('Error sending message:', error);
            // Add error message to history, marked as an error
            addMessageToHistory('assistant', `Error: ${error.message}`, true);
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

     // Initial focus
     // Add AI prefix to initial message in history display
     const initialAIMessage = historyDiv.querySelector('.server-message');
     if(initialAIMessage) initialAIMessage.textContent = "AI: Hello! How can I help you today?";
     // Add initial message to conversationHistory array?
     // Let's assume the initial greeting doesn't need to be part of the context sent back yet.
     
     messageInput.focus();
}); 