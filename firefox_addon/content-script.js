function get_src(e) {
    let img = document.querySelector(".input_box + img");
    let components = img["src"].split("/");
    return components[components.length - 1];
}

const BASE_URL = "https://7s7izxcb32.execute-api.us-east-1.amazonaws.com/decaptcha";

function decaptcha(e) {
    let src = get_src(e);
    let input_box = document.querySelector(".input_box[name=passwd2]");
    console.log(input_box);
    console.log(src);

    fetch(BASE_URL, {
        body: JSON.stringify({src: src}),
        headers: {
            "Content-Type": "application/json",
        },
        method: "POST",
    })
        .then(resp => resp.json())
        .then(json => input_box["value"] = json['answer']);
}

decaptcha();
