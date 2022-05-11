const express = require("express");
const secrets = require("./secrets.json");
const nodemailer = require("nodemailer");
var fs = require("fs");

const app = express();
app.use(express.json({ limit: 10000000 }));
const MongoClient = require("mongodb").MongoClient;
const connectionString = `mongodb://${secrets.mongodb_user}:${secrets.mongodb_pw}@localhost:27017`;

const sessionQuestionnaireEmail = fs.readFileSync(
  "session_questionnaire_email.txt",
  "utf8"
);

MongoClient.connect(connectionString, { useUnifiedTopology: true }).then(
  (client) => {
    console.log("Connected to Database");
    const db = client.db("log");
    const logCollection = db.collection("logging");
    const emailCollection = db.collection("email");

    app.listen(50000, () => {
      console.log("listening on 50000");
    });

    setInterval(() => {
      emailRunner(logCollection, emailCollection);
    }, 60000);

    app
      .route("/")
      .get((req, res) => {
        res.send("Logging server is running!");
        // sendMail("jacob-z@live.de", "test", emailCollection);
      })
      .post((req, res) => {
        logCollection
          .insertMany(req.body)
          .then((result) => {
            res.send("OK");
          })
          .catch((error) => console.error(error));
      });
  }
);

async function emailRunner(logCollection, emailCollection) {
  console.log("Checking whether to send emails...")

  // get list of users who logged more than 30 items in the last 24 hours, but not in the last 1 hour
  const lastDay = new Date();
  lastDay.setHours(lastDay.getHours() - 24);

  const lastHour = new Date();
  lastHour.setHours(lastHour.getHours() - 1);

  let users = await logCollection
    .aggregate([
      {
        $match: {
          time: { $gt: lastDay.toISOString() },
        },
      },
      {
        $group: {
          _id: "$uid",
          last_date: {
            $last: "$time",
          },
          num_entries: {
            $sum: 1,
          },
          email: {
            $first: "$data.email",
          },
        },
      },
      {
        $match: {
          last_date: { $lt: lastHour.toISOString() },
          num_entries: { $gt: 30 },
        },
      },
    ])
    .toArray();
  console.log("Active users:", users);

  // get list of users to whom an email has already been sent within the last 24 hours
  let alreadySentEmails = await emailCollection
    .aggregate([
      {
        $match: {
          date: { $gt: lastDay.toISOString() },
        }        
      },
      {
        $group: {
          _id: "$email"
        }
      }
    ])
    .toArray();
  alreadySentEmails = alreadySentEmails.map(o => o._id);
  console.log("Already sent to: ", alreadySentEmails);


  // send email to users except those who have already received on in the last 24 hours
  for(const user of users) {
    if (!alreadySentEmails.includes(user.email)) {
      sendMail(user.email, `https://jacobz.limesurvey.net/238976?newtest=Y&uid=${user._id}&email=${user.email}`, emailCollection)
    }
  }
}


function sendMail(receiverMail, studyUrl, emailCollection) {
  console.log("Sending mail...");
  let transporter = nodemailer.createTransport({
    host: secrets.smtp_server,
    port: 587,
    secure: false,
    auth: {
      user: secrets.smtp_user,
      pass: secrets.smtp_pass,
    },
  });

  transporter.sendMail(
    {
      from: `"${secrets.smtp_username}" <${secrets.smtp_user}>`, // sender address
      to: receiverMail, // list of receivers
      subject: "NornsVAE Session Questionnaire", // Subject line
      text: sessionQuestionnaireEmail.replace("%url", studyUrl), // plain text body
      // html: "<b>Hello world?</b>", // html body
    },
    (error, info) => {
      if (error) {
        console.log(error);
      }
      if (info) {
        console.log("Message sent to %s: %s", receiverMail, info.messageId);
      }

      emailCollection
        .insertOne({
          email: receiverMail,
          date: new Date().toISOString(),
          info,
          error,
        })
        .then((result) => {
          console.log(result);
        })
        .catch((error) => console.error(error));
    }
  );
}
