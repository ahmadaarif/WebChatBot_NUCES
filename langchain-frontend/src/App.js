import React from 'react';
import LanguageTranslator from './LanguageTranslator';
import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <h1>Langchain Translator</h1>
        <LanguageTranslator />
        
      </header>
    </div>
  );
}

export default App;
