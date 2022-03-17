const express = require('express');
const app = express();
app.use(express.json());
const MongoClient = require('mongodb').MongoClient
const connectionString = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false';

MongoClient.connect(connectionString, { useUnifiedTopology: true })
  .then(client => {
    console.log('Connected to Database')
    const db = client.db('log')
    const logCollection = db.collection('Logging')
    
    app.listen(3000, function() {
      console.log('listening on 3000')
    })

    app.get('/', function(req, res) {
      res.send('Logging server is running!')
    })

    app.post('/log', (req, res) => {
      logCollection.insertMany(req.body)
        .then(result => {
          console.log(result)
          res.send(req.body)
        })
        .catch(error => console.error(error))
    })
  })
