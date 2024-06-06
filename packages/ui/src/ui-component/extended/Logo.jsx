// import logo from '@/assets/images/flowise_logo.png'
import logo from '@/assets/images/deepthinking_logo.png'
import logoDark from '@/assets/images/flowise_logo_dark.png'

import { useSelector } from 'react-redux'

// ==============================|| LOGO ||============================== //

const Logo = () => {
    const customization = useSelector((state) => state.customization)

    return (
        <div style={{ alignItems: 'center', display: 'flex', flexDirection: 'row' }}>
            <img
                style={{ objectFit: 'contain', height: 'auto', width: 46 }}
                src={customization.isDarkMode ? logoDark : logo}
                alt='Flowise'
            />
            <span style={{ marginLeft: '16px', fontSize: '18px', fontWeight: 'bold' }}>深思大模型</span>
        </div>
    )
}

export default Logo
