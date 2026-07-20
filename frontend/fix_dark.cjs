const fs = require('fs');
const path = require('path');

const walkSync = function(dir, filelist) {
  files = fs.readdirSync(dir);
  filelist = filelist || [];
  files.forEach(function(file) {
    if (fs.statSync(path.join(dir, file)).isDirectory()) {
      filelist = walkSync(path.join(dir, file), filelist);
    }
    else {
      if (file.endsWith('.jsx')) {
        filelist.push(path.join(dir, file));
      }
    }
  });
  return filelist;
};

const componentsPaths = walkSync('src');

componentsPaths.forEach(file => {
  let content = fs.readFileSync(file, 'utf8');
  // Regex to remove dark:[anything] up to space, quote, or backtick
  const newContent = content.replace(/dark:[^\s'"`]+/g, '');
  fs.writeFileSync(file, newContent, 'utf8');
});

console.log("Removed all dark mode classes from components.");
