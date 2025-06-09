function addToCart(itemId, itemName, itemPrice, quantity = 1) {
    fetch('/check_login_status', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (!data.logged_in) {
            alert('Please log in first to add items to your cart');
            window.location.href = '/login';
            return;
        }
        
        fetch('/add_to_cart', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `item_id=${itemId}&quantity=${quantity}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const cartCountElem = document.querySelector('.cart-count');
                if (cartCountElem) {
                    cartCountElem.textContent = data.cart_count;
                }
                
                alert(`${itemName} has been added to your cart`);
            } else {
                alert('Failed to add item: ' + (data.message || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred, please try again');
        });
    })
    .catch(error => {
        console.error('Error checking login status:', error);
        alert('Error checking login status, please try again');
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const removeButtons = document.querySelectorAll('.remove-item');
    if (removeButtons) {
        removeButtons.forEach(button => {
            button.addEventListener('click', function() {
                const itemId = this.getAttribute('data-id');
                
                fetch('/remove_from_cart', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ item_id: itemId })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    } else {
                        alert('Failed to remove item');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred, please try again');
                });
            });
        });
    }
});