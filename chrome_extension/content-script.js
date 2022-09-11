const CCXP_URL = "https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/";
const SERVER_URL = "https://7s7izxcb32.execute-api.us-east-1.amazonaws.com/decaptcha";

function get_src(e) {
    let img = document.querySelector(".input_box + img");
    if (img === null) {
        return null;
    }

    let components = img["src"].split("/");
    return components[components.length - 1];
}

function download_img(src) {
    return fetch(CCXP_URL + src)
        .then(resp => resp.blob())
        .then(blob => new Promise((resolve, reject) => {
            let reader = new FileReader();
            reader.onload = () => {
                let img = reader.result;
                img = new Uint8Array(img);
                img = new Array(...img);
                resolve(img)
            };
            reader.onerror = reject;
            reader.readAsArrayBuffer(blob);
        }));
}

function decaptcha(e) {
    let src = get_src(e);
    if (src === null) {
        return;
    }

    let input_box = document.querySelector(".input_box[name=passwd2]");
    download_img(src)
        .then(img => {
            fetch(SERVER_URL, {
                body: JSON.stringify({src: src, img: img}),
                headers: {
                    "Content-Type": "application/json",
                },
                method: "POST",
            })
                .then(resp => resp.json())
                .then(json => input_box["value"] = json['answer']);
        });
}

addEventListener('load', decaptcha)
