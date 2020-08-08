import React, { Component } from 'react';
import ParticlesBg from 'particles-bg';

class App extends Component {

  async postData() {
    try {
      console.log(this.event);
      // implement post functionality

      // sample json result
      var myjsonstring = '{ "key" : "fileId", "class": "10 rupee coin" }';
      this.readJson(myjsonstring);
    }
    catch(e) {
      console.log(e);
    }
  }

  readJson(myjsonstring) {
    var classJson = JSON.parse(myjsonstring);
    this.readOutAloud(classJson);
  }

  readOutAloud(classJson) {
    var msg = new SpeechSynthesisUtterance();
    msg.text = classJson.class;
    window.speechSynthesis.speak(msg);
  }

  selectImage() {
    const realFileBtn = document.getElementById("realFile");
    realFileBtn.click();
  }

  onSelect() {
    const realFileBtn = document.getElementById("realFile");
    const customButton = document.getElementById("customButton");
    var imagePreview = document.getElementById("imagePreview");
    if (realFileBtn.value) {
      customButton.innerHTML = realFileBtn.value.match(
        /[\/\\]([\w\d\s\.\-\(\)]+)$/
      )[1];
    } else {
      customButton.innerHTML = "Select Image";
    }
    
  }

  render() {
    return (
      <div className="App">
        <div id="divBody">
          <input type="file" id="realFile" hidden="hidden" onChange={ () => this.onSelect() }/>
          <button type="button" id="customButton" onClick={ () => this.selectImage() }>Select Image</button>
          <br/>
          <img src="" alt="" id="imagePreview"/>
          <br/>
          <button type="button" id="customButtonSubmit" onClick={ () => this.postData() }>submit</button>
        </div>    
        <ParticlesBg type="circle" bg={true} />    
      </div>
    );
  }
}

export default App;