document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message');
    const sendButton = document.getElementById('send');
    const historyDiv = document.getElementById('history');
    const serverUrl = '/chat'; // Endpoint defined in server.cpp

    let conversationHistory = []; // Store history as {role: 'user'/'assistant', content: '...'} 

    function addMessageToHistory(role, text, isError = false) {
        if (!isError) {  // error banners must not enter the model's prompt
            conversationHistory.push({ role: role, content: text });
        }
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

    // Build a multi-turn prompt from recent history. The word-level models
    // have a 128-word context window, so keep the prompt to a sliding window
    // of the most recent turns under a word budget (leaving room to reply).
    function buildPrompt(latestUserText) {
        const WORD_BUDGET = 80;
        const turns = conversationHistory
            .map(t => (t.role === 'user' ? 'user: ' : 'assistant: ') + t.content)
            .concat(['user: ' + latestUserText, 'assistant:']);
        // Take turns from the end until the budget is filled
        let words = 0;
        const kept = [];
        for (let i = turns.length - 1; i >= 0; i--) {
            const w = turns[i].split(/\s+/).length;
            if (words + w > WORD_BUDGET && kept.length > 1) break;
            kept.unshift(turns[i]);
            words += w;
        }
        return kept.join(' ');
    }

    async function sendMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText) return; // Don't send empty messages

        const prompt = buildPrompt(messageText);
        addMessageToHistory('user', messageText);
        messageInput.value = ''; // Clear input field

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_input: prompt,
                    temperature: 0.7,  // Tiny models at 0.1 collapse to their most generic phrases
                    max_new_tokens: 45,
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