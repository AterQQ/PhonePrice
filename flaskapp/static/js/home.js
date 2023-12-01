document.addEventListener('DOMContentLoaded', function () {
    const inputList = [
        document.querySelector('#battery_power'),
        document.querySelector('#clock_speed'),
        document.querySelector('#fc'),
        document.querySelector('#pc'),
        document.querySelector('#int_memory'),
        document.querySelector('#n_cores'),
        document.querySelector('#ram'),
        document.querySelector('#m_dep'),
        document.querySelector('#mobile_wt'),
        document.querySelector('#px_height'),
        document.querySelector('#px_width'),
        document.querySelector('#sc_h'),
        document.querySelector('#sc_w'),
        document.querySelector('#talk_time'),
    ];
    
    const outputList = [
        document.querySelector('#battery_output'),
        document.querySelector('#clock_output'),
        document.querySelector('#fc_output'),
        document.querySelector('#pc_output'),
        document.querySelector('#memory_output'),
        document.querySelector('#cores_output'),
        document.querySelector('#ram_output'),
        document.querySelector('#m_dep_output'),
        document.querySelector('#mobile_wt_output'),
        document.querySelector('#px_height_output'),
        document.querySelector('#px_width_output'),
        document.querySelector('#sc_h_output'),
        document.querySelector('#sc_w_output'),
        document.querySelector('#talk_output'),
    ];
    
    for (let i = 0; i < inputList.length; i++) {
        let input = inputList[i]
        let output = outputList[i]
        output.textContent = input.value;
    
        input.addEventListener("input", (event) => {
            output.textContent = event.target.value;
        });
    }
});

function submitForm() {
    document.getElementById("inputForm").submit();
}