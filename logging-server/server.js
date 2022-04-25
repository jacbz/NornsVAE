const express = require("express");
const secrets = require("./secrets.json");

const app = express();
app.use(express.json());
const MongoClient = require("mongodb").MongoClient;
const connectionString = `mongodb://${secrets.mongodb_user}:${secrets.mongodb_pw}@localhost:27017`;

MongoClient.connect(connectionString, { useUnifiedTopology: true }).then(
  (client) => {
    console.log("Connected to Database");
    const db = client.db("log");
    const logCollection = db.collection("logging");

    app.listen(50000, () => {
      console.log("listening on 50000");
    });

    app
      .route("/")
      .get((req, res) => {
        res.send("Logging server is running!");
      })
      .post((req, res) => {
        logCollection
          .insertMany(req.body)
          .then((result) => {
            console.log(result);
            res.send(req.body);
          })
          .catch((error) => console.error(error));
      });
  }
);
