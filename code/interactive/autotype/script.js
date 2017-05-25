(function () {

function fetchData(cb) {
  const xhr = new XMLHttpRequest();
  xhr.addEventListener("load", function () {
    const translations = xhr.responseText.split('\n');
    translations.pop();
    cb(null, translations.map(JSON.parse));
  });
  xhr.open('get', '/data.json.newline');
  xhr.send();
}

function Show() {
  this.source = document.querySelector('#source .text');
  this.target = document.querySelector('#target .text');
  console.log(this.target);
  this.translation = document.querySelector('#translation .text');
  this.bleu = document.querySelector('#bleu .text');
}

Show.prototype.animateText = function (element, text, cb) {
  const self = this;
  const tokens = text.split(' ');

  element.innerHTML = '';

  (function recursive() {
    if (tokens.length === 0) return cb(null);
    self.set(element, element.innerHTML + tokens.shift() + ' ');

    setTimeout(recursive, 30);
  })();
}

Show.prototype.clear = function (element) {
  element.classList.add('clear');
};

Show.prototype.set = function (element, text) {
  element.classList.remove('clear');
  element.innerHTML = text;
};

Show.prototype.update = function (pair, done) {
  const self = this;
  self.clear(self.source);
  self.clear(self.target);
  self.clear(self.translation);
  self.clear(self.bleu);

  self.animateText(this.source, pair.source, function () {
    setTimeout(function () {
      self.set(self.target, pair.target);
      setTimeout(function () {
        self.set(self.translation, pair.translation);
        self.set(self.bleu, pair.bleu.toFixed(2) + '%');
        setTimeout(function () {
          done(null);
        }, pair.target.split(' ').length * 1000);
      }, 1000);
    }, 100);
  });
};

function Queue(updater) {
  this.updater = updater;
  this.queue = [];
  this.draining = false;
}

Queue.prototype.drain = function () {
  const self = this;

  if (this.draining) return;
  if (this.queue.length == 0) return;

  this.draining = true;
  const value = this.queue.shift();

  this.updater.update(value, function (err) {
    self.draining = false;
    self.drain();
  });
};

Queue.prototype.push = function (value) {
  this.queue.push(value);
  if (!this.draining) {
      this.drain();
  }
};

document.addEventListener('DOMContentLoaded', function () {
  const show = new Show();
  const queue = new Queue(show);

  fetchData(function (err, data) {
    const sorted = data.sort(function (a, b) {
      return b.bleu - a.bleu;
    }).filter(function (v) {
      return (v.target.split(' ').length > 15 && v.source.length < 180);
    });

    for (let i = 0; i < 100; i++) {
      const index = Math.floor(Math.random() * 300);
      queue.push(sorted[index]);
      sorted.splice(index, 1);
    }


  });
});

})();
