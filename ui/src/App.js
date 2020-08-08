import React, { useState } from 'react';
import ParticlesBg from 'particles-bg';

function App() {

    const [ image, setImage ] = useState();

    const postData = () => {
        try {
            var msg = new SpeechSynthesisUtterance();
            msg.text = "image selected. Analysing coin image";
            window.speechSynthesis.speak(msg);

            const uploadData = new FormData();
            uploadData.append('picture', image);

            fetch("http://localhost:8000/coinlist/", {
              method: 'POST',
              body: uploadData
            })
            .then(response => response.json()) 
  
            // Displaying results to console 
              .then(function(json){
                var result;
                  if (json === 3) {
                    result = "1 rupee";
                  }
                  else if (json === 1) {
                    result = "2 rupee";
                  }
                  else if (json === 0) {
                    result = "5 rupee";
                  }
                  else if (json === 3) {
                    result = "10 rupee";
                  }

                  msg.text = result;
                  window.speechSynthesis.speak(msg);
                });
            
          }
          catch(e) {
            console.log(e);
          }
    }

    return (
      <div className="App">
        <div id="divBody">
          <input type="file" id="realFile" onChange={ (event) => setImage(event.target.files[0]) }/>
          <br/>
          <button type="button" id="customButtonSubmit" onClick={ () => postData() }>submit</button>
        </div>    
        <ParticlesBg type="circle" bg={true} />    
      </div>
    );
}

export default App;