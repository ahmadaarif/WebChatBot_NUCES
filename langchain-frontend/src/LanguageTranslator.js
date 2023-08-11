import React, { useState } from 'react';
import axios from 'axios';

const LanguageTranslator = () => {
    const [query, setQuery] = useState(' ');
    const [result, setResult] = useState(' ');

    const handleQuerySubmmit = async () => {
        try {
            const response = await axios.get ('/process_query/', { query });
            setResult(response.data.result);
        }
        catch (error) {
            console.error('Error processing query: ', error);
        }
    };

    return (
        <div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={handleQuerySubmmit}>Submit query</button>
      <div>
        <p>Result:</p>
        <p>{result}</p>
      </div>
    </div>
  );
};

export default LanguageTranslator;